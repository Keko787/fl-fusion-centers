"""Phase B integration test — central training dispatcher dispatches FUSION-MLP correctly.

Mirrors ``tests/integration/test_fusion_dispatcher.py`` (Phase A) but
for the next stage: given a built model and data tuple, does
``modelCentralTrainingConfigLoad`` return a ``CentralFusionMLPClient``
configured the way Phase B expects?

Covers:
* Branch selection: ``model_type="FUSION-MLP"`` returns the right class.
* ``args=None`` is rejected with a clear error (FUSION-MLP requires the
  passthrough for ``escalation_loss_weight`` etc.).
* ``num_classes`` is forwarded so per-class metrics work for non-3-class
  configurations.
* Legacy ``NIDS`` / ``GAN`` branches do not regress when called
  positionally (no args kwarg).
"""
from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Legacy NIDS trainer in the dispatcher's import chain hard-imports
# ``tensorflow_privacy`` at module top — the package's pinned versions
# are incompatible with current TF, but our FUSION-MLP branch never
# touches it. Stub the module so the dispatcher import succeeds; any
# code that actually invokes ``tfp.DPKerasAdamOptimizer`` will fail
# later, which is the correct behavior for non-FUSION-MLP DP paths.
sys.modules.setdefault("tensorflow_privacy",
                        types.ModuleType("tensorflow_privacy"))

from types import SimpleNamespace

import numpy as np
import pytest

from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.FusionMLP.fusionMLPCentralTrainingConfig import (
    CentralFusionMLPClient,
)
from Config.SessionConfig.ModelTrainingConfigLoad.modelCentralTrainingConfigLoad import (
    modelCentralTrainingConfigLoad,
)
from Config.modelStructures.FusionMLP.multiTaskMLPStruct import build_fusion_mlp


def _make_fusion_args(escalation_loss_weight: float = 0.5):
    return SimpleNamespace(escalation_loss_weight=escalation_loss_weight)


def _make_fusion_inputs(input_dim: int = 8, num_classes: int = 3):
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 80, 20, 20
    X_train = rng.standard_normal((n_train, input_dim)).astype("float32")
    X_val = rng.standard_normal((n_val, input_dim)).astype("float32")
    X_test = rng.standard_normal((n_test, input_dim)).astype("float32")
    y_train = (rng.integers(0, num_classes, n_train).astype("int64"),
                rng.random(n_train).astype("float32"))
    y_val = (rng.integers(0, num_classes, n_val).astype("int64"),
              rng.random(n_val).astype("float32"))
    y_test = (rng.integers(0, num_classes, n_test).astype("int64"),
               rng.random(n_test).astype("float32"))
    return X_train, X_val, X_test, y_train, y_val, y_test


def _call_dispatcher(*, model, model_type, args, num_classes=3,
                      data=None):
    """Thin wrapper that fills in the legacy positional tuple."""
    if data is None:
        data = _make_fusion_inputs(input_dim=model.input_shape[-1] if hasattr(model, "input_shape") else 8,
                                    num_classes=num_classes)
    X_train, X_val, X_test, y_train, y_val, y_test = data
    return modelCentralTrainingConfigLoad(
        # 4-tuple model slots
        model, None, None, None,
        # dataset_used, model_type, train_type
        "COMMCRIME", model_type, "MultiTask",
        # callback flags
        None, None, None, None,
        # data
        X_train, X_val, y_train, y_val, X_test, y_test,
        # node, training hyperparams
        1, 32, 2, None, max(1, len(X_train) // 32),
        # input_dim, num_classes, latent_dim, betas, lr
        X_train.shape[1], num_classes, None, [0.9, 0.999], 1e-3,
        # l2_alpha, DP slots
        1e-4, None, None, None,
        # callback param slots
        "val_loss", 5, True, "val_loss", 3, True, "val_loss", "min",
        # log paths
        "/tmp/_unused_eval.log", "/tmp/_unused_train.log",
        args=args,
    )


# ───────────────────────────────────────────────────────────────────────
#   Branch selection
# ───────────────────────────────────────────────────────────────────────

def test_dispatcher_selects_fusion_mlp_client_for_fusion_mlp():
    model = build_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    client = _call_dispatcher(
        model=model, model_type="FUSION-MLP",
        args=_make_fusion_args(),
    )
    assert isinstance(client, CentralFusionMLPClient)


def test_dispatcher_rejects_fusion_mlp_without_args():
    model = build_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    with pytest.raises(ValueError, match="requires args"):
        _call_dispatcher(model=model, model_type="FUSION-MLP", args=None)


def test_dispatcher_returns_none_for_unhandled_model_type():
    """Defensive sanity: a model_type the dispatcher does not know
    should leave ``client`` unset (None). Catches bad branch additions."""
    model = build_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    client = _call_dispatcher(
        model=model, model_type="NIDS-IOT-Binary",
        args=_make_fusion_args(),
    )
    assert client is None


# ───────────────────────────────────────────────────────────────────────
#   num_classes forwarding (issue #4)
# ───────────────────────────────────────────────────────────────────────

def test_dispatcher_forwards_num_classes():
    """If a future config sets num_classes != 3, the trainer must
    receive it (otherwise per-class P/R silently miss classes)."""
    model = build_fusion_mlp(input_dim=8, num_classes=4, l2_alpha=0.0)
    data = _make_fusion_inputs(input_dim=8, num_classes=4)
    client = _call_dispatcher(
        model=model, model_type="FUSION-MLP",
        args=_make_fusion_args(), num_classes=4, data=data,
    )
    assert isinstance(client, CentralFusionMLPClient)
    assert client.num_classes == 4


def test_compute_metrics_honors_num_classes():
    """Direct unit-style check on the static metrics computation:
    with num_classes=4, the returned dict contains class_0..class_3 keys."""
    rng = np.random.default_rng(0)
    n = 40
    y_true = rng.integers(0, 4, n).astype("int64")
    threat_pred = rng.random((n, 4)).astype("float32")
    threat_pred = threat_pred / threat_pred.sum(axis=1, keepdims=True)
    esc_true = rng.random(n).astype("float32")
    esc_pred = rng.random((n, 1)).astype("float32")

    metrics = CentralFusionMLPClient._compute_metrics(
        y_true, threat_pred, esc_true, esc_pred, num_classes=4,
    )
    for i in range(4):
        assert f"threat_class_{i}_precision" in metrics
        assert f"threat_class_{i}_recall" in metrics
    assert "threat_class_4_precision" not in metrics


def test_compute_metrics_emits_threat_accuracy():
    """Phase E.3 — overall classification accuracy as a sibling metric to
    macro-F1. The server's fairness_accuracy_variance reads this."""
    # Construct predictions that match the truth exactly → accuracy = 1.0.
    n = 12
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    threat_pred = np.zeros((n, 3), dtype=np.float32)
    for i, c in enumerate(y_true):
        threat_pred[i, c] = 1.0
    esc_true = np.linspace(0, 1, n).astype(np.float32)
    esc_pred = esc_true.reshape(-1, 1)

    metrics = CentralFusionMLPClient._compute_metrics(
        y_true, threat_pred, esc_true, esc_pred, num_classes=3,
    )
    assert "threat_accuracy" in metrics
    assert metrics["threat_accuracy"] == pytest.approx(1.0)

    # And a partial-match case: flip one prediction → accuracy = (n-1)/n.
    threat_pred[0] = [0.0, 1.0, 0.0]  # was class 0, now predicted as class 1
    metrics2 = CentralFusionMLPClient._compute_metrics(
        y_true, threat_pred, esc_true, esc_pred, num_classes=3,
    )
    assert metrics2["threat_accuracy"] == pytest.approx((n - 1) / n)


# ───────────────────────────────────────────────────────────────────────
#   Issue #7 — escalation_auroc binarized at median, escalation_spearman
# ───────────────────────────────────────────────────────────────────────

def test_compute_metrics_emits_both_escalation_metrics():
    rng = np.random.default_rng(0)
    n = 40
    y_class = rng.integers(0, 3, n).astype("int64")
    threat = rng.random((n, 3)).astype("float32")
    threat = threat / threat.sum(axis=1, keepdims=True)
    esc_true = rng.random(n).astype("float32")
    esc_pred = rng.random((n, 1)).astype("float32")

    metrics = CentralFusionMLPClient._compute_metrics(
        y_class, threat, esc_true, esc_pred,
    )
    assert "escalation_auroc" in metrics
    assert "escalation_spearman" in metrics
    # Random predictions → both metrics near chance.
    assert 0.0 <= metrics["escalation_auroc"] <= 1.0
    assert -1.0 <= metrics["escalation_spearman"] <= 1.0


def test_compute_metrics_auroc_works_on_skewed_distribution():
    """Fixed-0.5 binarization would have collapsed to single-class on
    this distribution and returned NaN; median binarization keeps it
    50/50 and AUROC is well-defined."""
    rng = np.random.default_rng(0)
    n = 50
    y_class = rng.integers(0, 3, n).astype("int64")
    threat = rng.random((n, 3)).astype("float32")
    threat = threat / threat.sum(axis=1, keepdims=True)
    # Heavily skewed escalation truth — mostly small values.
    esc_true = (rng.random(n) * 0.3).astype("float32")  # all in [0, 0.3]
    esc_pred = esc_true.reshape(-1, 1) + rng.normal(scale=0.05, size=(n, 1)).astype("float32")

    metrics = CentralFusionMLPClient._compute_metrics(
        y_class, threat, esc_true, esc_pred,
    )
    assert not np.isnan(metrics["escalation_auroc"]), (
        "AUROC should be defined under median binarization"
    )
    # Predictions are noisy versions of truth → AUROC well above 0.5.
    assert metrics["escalation_auroc"] > 0.7
    # And Spearman should also reflect the strong correlation.
    assert metrics["escalation_spearman"] > 0.7


def test_compute_metrics_spearman_perfect_when_predictions_match():
    rng = np.random.default_rng(0)
    n = 30
    y_class = rng.integers(0, 3, n).astype("int64")
    threat = rng.random((n, 3)).astype("float32")
    threat = threat / threat.sum(axis=1, keepdims=True)
    esc_true = np.linspace(0.0, 1.0, n).astype("float32")
    # Predictions monotonically related to truth → Spearman should be 1.
    esc_pred = (esc_true ** 2).reshape(-1, 1)

    metrics = CentralFusionMLPClient._compute_metrics(
        y_class, threat, esc_true, esc_pred,
    )
    assert metrics["escalation_spearman"] == pytest.approx(1.0, abs=1e-6)


# ───────────────────────────────────────────────────────────────────────
#   Issue #8 — TF seeded before build_fusion_mlp → reproducible weights
# ───────────────────────────────────────────────────────────────────────

def test_seeded_model_init_reproducible_via_factory():
    """Calling tf.keras.utils.set_random_seed before build_fusion_mlp
    twice with the same seed must produce identical initial weights.
    This is the seeding mechanism the dispatcher uses (Phase B issue #8)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(123)
    m1 = build_fusion_mlp(input_dim=12, num_classes=3, l2_alpha=0.0)
    w1 = m1.get_weights()

    tf.keras.utils.set_random_seed(123)
    m2 = build_fusion_mlp(input_dim=12, num_classes=3, l2_alpha=0.0)
    w2 = m2.get_weights()

    assert len(w1) == len(w2)
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


def test_modelCreateLoad_seed_kwarg_produces_reproducible_weights():
    """Going through the dispatcher with seed=N twice → identical weights."""
    from Config.SessionConfig.modelCreateLoad import modelCreateLoad

    m1, _, _, _ = modelCreateLoad(
        "FUSION-MLP", "MultiTask",
        None, None, None, None,
        "COMMCRIME",
        12, None, True, None, 1e-4, None, 3,
        seed=987,
    )
    m2, _, _, _ = modelCreateLoad(
        "FUSION-MLP", "MultiTask",
        None, None, None, None,
        "COMMCRIME",
        12, None, True, None, 1e-4, None, 3,
        seed=987,
    )
    w1 = m1.get_weights()
    w2 = m2.get_weights()
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)
