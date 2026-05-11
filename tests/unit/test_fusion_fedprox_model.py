"""Phase D.2 — FedProxFusionMLPModel proximal-term tests.

Pins the proximal math + the gated train_step behavior:
* ``set_global_weights`` stores the anchor correctly.
* Proximal term is zero when weights match the anchor.
* Proximal term grows quadratically with ``||w - g||``.
* With ``mu=0`` (or no anchor set), train_step matches the base
  Model — FedProx model behaves identically to FedAvg.
* With ``mu>0`` and anchor set, training pulls weights toward the
  anchor more than ``mu=0`` would.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from Config.modelStructures.FusionMLP.multiTaskMLPStruct import (
    FedProxFusionMLPModel,
    build_fedprox_fusion_mlp,
    build_fusion_mlp,
)


def _compile_model(model, lr=0.01):
    model.compile(
        optimizer=SGD(learning_rate=lr),
        loss={"threat": "sparse_categorical_crossentropy",
              "escalation": "binary_crossentropy"},
        loss_weights={"threat": 0.5, "escalation": 0.5},
    )


def _proximal_term(model) -> float:
    """Compute Σ ||w - g||² manually for verification."""
    return float(sum(
        np.sum(np.square(w.numpy() - g.numpy()))
        for w, g in zip(model.trainable_weights,
                         model.__dict__["_global_weight_vars"])
    ))


# ───────────────────────────────────────────────────────────────────────
#   Construction / anchoring
# ───────────────────────────────────────────────────────────────────────

def test_build_fedprox_returns_subclass():
    model = build_fedprox_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    assert isinstance(model, FedProxFusionMLPModel)
    assert model.fedprox_mu == 0.0
    assert model.__dict__["_global_weight_vars"] is None
    assert not model._global_weights_set


def test_set_global_weights_stores_anchor():
    """``set_global_weights`` takes the FULL weights list (matching
    ``get_weights()``) and filters to the trainable subset internally."""
    model = build_fedprox_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    full = model.get_weights()
    model.set_global_weights(full)
    assert model._global_weights_set

    # Anchors align element-wise with trainable_weights, not weights.
    trainable_initial = [w.numpy() for w in model.trainable_weights]
    for g, expected in zip(model.__dict__["_global_weight_vars"],
                            trainable_initial):
        np.testing.assert_array_equal(g.numpy(), expected)


def test_set_global_weights_updates_on_second_call():
    """Per-round contract: anchor changes when the server broadcasts new
    weights, but the underlying variables are re-used (no reallocation)."""
    model = build_fedprox_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    first_full = model.get_weights()
    model.set_global_weights(first_full)
    var_id = id(model.__dict__["_global_weight_vars"][0])

    second_full = [w + 1.0 for w in first_full]
    model.set_global_weights(second_full)
    assert id(model.__dict__["_global_weight_vars"][0]) == var_id  # same Variable, .assign

    # First trainable anchor should reflect the new value.
    trainable_indices = model.__dict__["_trainable_indices"]
    expected_first_anchor = second_full[trainable_indices[0]]
    np.testing.assert_array_equal(
        model.__dict__["_global_weight_vars"][0].numpy(),
        expected_first_anchor,
    )


def test_set_global_weights_rejects_wrong_length():
    """Length validation (review #9) — passing trainable-only list raises."""
    model = build_fedprox_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    trainable_only = [w.numpy() for w in model.trainable_weights]
    with pytest.raises(ValueError, match="expected"):
        model.set_global_weights(trainable_only)


# ───────────────────────────────────────────────────────────────────────
#   Review #6 — anchor variables don't leak into model.weights
# ───────────────────────────────────────────────────────────────────────

def test_global_anchors_do_not_pollute_model_weights():
    """``_global_weight_vars`` is stored via ``__dict__`` to bypass
    Keras's variable tracker. After ``set_global_weights``, the model's
    ``weights`` / ``get_weights()`` length must stay constant — otherwise
    the .keras save artifact would bloat with the anchor variables.
    """
    model = build_fedprox_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    before = len(model.weights)
    weights_before = len(model.get_weights())

    model.set_global_weights(model.get_weights())

    assert len(model.weights) == before, (
        f"anchor vars leaked into model.weights: {len(model.weights)} vs {before}"
    )
    assert len(model.get_weights()) == weights_before


# ───────────────────────────────────────────────────────────────────────
#   Proximal-term math
# ───────────────────────────────────────────────────────────────────────

def test_proximal_zero_when_weights_match_anchor():
    model = build_fedprox_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    model.set_global_weights(model.get_weights())
    assert _proximal_term(model) == pytest.approx(0.0, abs=1e-7)


def test_proximal_quadratic_in_perturbation():
    """Proximal term should grow as eps² when weights drift by eps.

    Assigns directly to ``trainable_weights`` rather than calling
    ``set_weights`` so we only perturb the weights the proximal term
    actually touches.
    """
    model = build_fedprox_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    initial_trainable = [w.numpy().copy() for w in model.trainable_weights]
    model.set_global_weights(model.get_weights())

    proximals = {}
    for eps in (0.1, 0.2, 0.4):
        for w, init in zip(model.trainable_weights, initial_trainable):
            w.assign(init + eps)
        proximals[eps] = _proximal_term(model)

    # Quadratic scaling: doubling eps → quadrupling proximal.
    assert proximals[0.2] / proximals[0.1] == pytest.approx(4.0, rel=0.01)
    assert proximals[0.4] / proximals[0.1] == pytest.approx(16.0, rel=0.01)


# ───────────────────────────────────────────────────────────────────────
#   train_step gating
# ───────────────────────────────────────────────────────────────────────

def test_fedprox_with_mu_zero_matches_base_after_one_step():
    """mu=0 → FedProx train_step takes the SAME gradient step as base
    Model. With matching seeds and identical compile, weights end up
    identical."""
    tf.keras.utils.set_random_seed(42)
    base = build_fusion_mlp(input_dim=4, num_classes=3, l2_alpha=0.0)
    tf.keras.utils.set_random_seed(42)
    fedprox = build_fedprox_fusion_mlp(input_dim=4, num_classes=3,
                                          l2_alpha=0.0, mu=0.0)

    _compile_model(base, lr=0.1)
    _compile_model(fedprox, lr=0.1)

    # Sync weights bit-for-bit before training.
    fedprox.set_weights(base.get_weights())

    x = np.random.RandomState(0).rand(8, 4).astype(np.float32)
    y = {"threat": np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64),
         "escalation": np.random.RandomState(1).rand(8).astype(np.float32)}

    base.fit(x, y, epochs=1, batch_size=8, verbose=0, shuffle=False)
    fedprox.fit(x, y, epochs=1, batch_size=8, verbose=0, shuffle=False)

    for b, f in zip(base.get_weights(), fedprox.get_weights()):
        np.testing.assert_allclose(b, f, atol=1e-5)


def test_fedprox_pulls_weights_toward_global_anchor():
    """With high mu and a clearly-separated anchor, FedProx training
    should leave weights CLOSER to the anchor than mu=0 training would,
    given the same starting point and the same data."""
    def _run(mu_value: float) -> float:
        tf.keras.utils.set_random_seed(123)
        m = build_fedprox_fusion_mlp(input_dim=4, num_classes=3,
                                       l2_alpha=0.0, mu=mu_value)
        _compile_model(m, lr=0.05)

        # Snapshot trainable weights for the anchor + post-training comparison.
        trainable_anchor = [w.numpy().copy() for w in m.trainable_weights]
        m.set_global_weights(m.get_weights())
        # Perturb starting point away from anchor.
        m.set_weights([w + 0.5 for w in m.get_weights()])

        x = np.random.RandomState(0).rand(16, 4).astype(np.float32)
        y = {"threat": np.random.RandomState(1).randint(0, 3, 16).astype(np.int64),
             "escalation": np.random.RandomState(2).rand(16).astype(np.float32)}
        m.fit(x, y, epochs=5, batch_size=8, verbose=0, shuffle=False)

        dist = sum(
            float(np.sum(np.square(w.numpy() - a)))
            for w, a in zip(m.trainable_weights, trainable_anchor)
        )
        return dist

    dist_high_mu = _run(10.0)
    dist_no_mu = _run(0.0)
    assert dist_high_mu < dist_no_mu, (
        f"FedProx with mu=10 didn't pull weights closer to anchor: "
        f"high_mu={dist_high_mu:.4f} vs no_mu={dist_no_mu:.4f}"
    )


# ───────────────────────────────────────────────────────────────────────
#   proximal_contribution metric (Phase D follow-up)
# ───────────────────────────────────────────────────────────────────────

def test_train_step_emits_proximal_contribution_in_history():
    """``history.history`` should include ``proximal_contribution`` —
    the sibling metric that exposes what the proximal term is doing
    (the plain ``loss`` metric excludes it per Keras's contract)."""
    tf.keras.utils.set_random_seed(31)
    model = build_fedprox_fusion_mlp(input_dim=4, num_classes=3,
                                       l2_alpha=0.0, mu=0.5)
    _compile_model(model, lr=0.01)
    model.set_global_weights(model.get_weights())

    # Perturb weights so proximal > 0
    for w in model.trainable_weights:
        w.assign(w.numpy() + 0.3)

    x = np.random.RandomState(0).rand(16, 4).astype(np.float32)
    y = {"threat": np.random.RandomState(1).randint(0, 3, 16).astype(np.int64),
         "escalation": np.random.RandomState(2).rand(16).astype(np.float32)}

    history = model.fit(x, y, epochs=2, batch_size=8, verbose=0, shuffle=False)
    assert "proximal_contribution" in history.history
    # First epoch's value should be clearly positive given mu=0.5 + 0.3
    # perturbation on every trainable weight.
    assert history.history["proximal_contribution"][0] > 0.0


def test_proximal_contribution_zero_when_anchor_not_set():
    """No ``set_global_weights`` call → proximal_contribution emitted as 0,
    not absent. Consistent schema across FedAvg / FedProx paths."""
    tf.keras.utils.set_random_seed(11)
    model = build_fedprox_fusion_mlp(input_dim=4, num_classes=3,
                                       l2_alpha=0.0, mu=0.5)
    _compile_model(model, lr=0.01)
    # NOTE: not calling set_global_weights — anchor stays None.

    x = np.random.RandomState(0).rand(8, 4).astype(np.float32)
    y = {"threat": np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64),
         "escalation": np.random.RandomState(1).rand(8).astype(np.float32)}

    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0, shuffle=False)
    assert "proximal_contribution" in history.history
    assert history.history["proximal_contribution"][0] == pytest.approx(0.0)


def test_proximal_contribution_zero_when_mu_is_zero():
    """mu=0 → proximal_contribution stays 0 even though anchors are set
    (the tf.where gate). Lets a FedAvg-style broadcast (no proximal)
    flow through the FedProx model without spurious metric values."""
    tf.keras.utils.set_random_seed(17)
    model = build_fedprox_fusion_mlp(input_dim=4, num_classes=3,
                                       l2_alpha=0.0, mu=0.0)
    _compile_model(model, lr=0.01)
    model.set_global_weights(model.get_weights())
    for w in model.trainable_weights:
        w.assign(w.numpy() + 0.3)  # perturb so Σ||w-g||² > 0

    x = np.random.RandomState(0).rand(8, 4).astype(np.float32)
    y = {"threat": np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64),
         "escalation": np.random.RandomState(1).rand(8).astype(np.float32)}

    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0, shuffle=False)
    # mu=0 → gate kills the contribution regardless of distance.
    assert history.history["proximal_contribution"][0] == pytest.approx(0.0)


def test_fedprox_no_anchor_set_behaves_like_base():
    """If set_global_weights was never called, the proximal term is
    inert even with mu > 0 (the gating ``self._global_weights_set``
    check skips it)."""
    tf.keras.utils.set_random_seed(7)
    base = build_fusion_mlp(input_dim=4, num_classes=3, l2_alpha=0.0)
    tf.keras.utils.set_random_seed(7)
    fedprox = build_fedprox_fusion_mlp(input_dim=4, num_classes=3,
                                          l2_alpha=0.0, mu=5.0)
    # NOTE: not calling set_global_weights on purpose.

    _compile_model(base, lr=0.05)
    _compile_model(fedprox, lr=0.05)
    fedprox.set_weights(base.get_weights())

    x = np.random.RandomState(0).rand(8, 4).astype(np.float32)
    y = {"threat": np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64),
         "escalation": np.random.RandomState(1).rand(8).astype(np.float32)}

    base.fit(x, y, epochs=1, batch_size=8, verbose=0, shuffle=False)
    fedprox.fit(x, y, epochs=1, batch_size=8, verbose=0, shuffle=False)

    for b, f in zip(base.get_weights(), fedprox.get_weights()):
        np.testing.assert_allclose(b, f, atol=1e-5)
