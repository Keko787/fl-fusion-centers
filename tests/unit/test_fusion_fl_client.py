"""Phase C.1 — federated client (FlFusionMLPClient) protocol tests.

Pins the Flower NumPyClient contract for FUSION-MLP:
* ``get_parameters(config)`` returns the model's current weights.
* ``fit(parameters, config)`` accepts a flat list of ndarrays, applies
  them, trains for one round, returns ``(new_weights, n_train, metrics)``.
* ``evaluate(parameters, config)`` returns ``(total_loss, n_test, metrics)``
  with the Phase B per-class metrics + escalation_auroc + escalation_spearman.
* Two consecutive fit() calls with the same starting parameters and
  same shuffle_seed produce identical results (FL reproducibility).
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

from Config.ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.FusionMLP.fusionMLPClientConfig import (
    FlFusionMLPClient,
)
from Config.modelStructures.FusionMLP.multiTaskMLPStruct import build_fusion_mlp


def _make_client(tmp_path, *, shuffle_seed=0, epochs=1,
                  print_summary_enabled=False):
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 80, 20, 20
    X_train = rng.standard_normal((n_train, 6)).astype("float32")
    X_val = rng.standard_normal((n_val, 6)).astype("float32")
    X_test = rng.standard_normal((n_test, 6)).astype("float32")
    y_train = (rng.integers(0, 3, n_train).astype("int64"),
                rng.random(n_train).astype("float32"))
    y_val = (rng.integers(0, 3, n_val).astype("int64"),
              rng.random(n_val).astype("float32"))
    y_test = (rng.integers(0, 3, n_test).astype("int64"),
               rng.random(n_test).astype("float32"))

    model = build_fusion_mlp(input_dim=6, num_classes=3, l2_alpha=0.0)
    return FlFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_val, x_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        BATCH_SIZE=16, epochs=epochs,
        steps_per_epoch=max(1, n_train // 16),
        learning_rate=1e-3,
        evaluation_log=str(tmp_path / "eval.log"),
        training_log=str(tmp_path / "train.log"),
        shuffle_seed=shuffle_seed,
        print_summary=print_summary_enabled,
    )


def test_get_parameters_returns_model_weights(tmp_path):
    client = _make_client(tmp_path)
    params = client.get_parameters(config={})
    expected = client.model.get_weights()
    assert len(params) == len(expected)
    for p, e in zip(params, expected):
        np.testing.assert_array_equal(p, e)


def test_fit_contract(tmp_path):
    client = _make_client(tmp_path)
    initial_params = client.get_parameters(config={})
    new_params, n_train, metrics = client.fit(initial_params, config={})

    # Shape contract
    assert len(new_params) == len(initial_params)
    for n, i in zip(new_params, initial_params):
        assert n.shape == i.shape
    assert n_train == 80

    # Metrics contract
    for key in ("loss", "val_loss",
                "threat_loss", "escalation_loss",
                "val_threat_loss", "val_escalation_loss"):
        assert key in metrics
        assert isinstance(metrics[key], float)

    # The weights should actually have moved after one training round.
    moved = any(
        not np.array_equal(n, i) for n, i in zip(new_params, initial_params)
    )
    assert moved, "weights unchanged after a fit() round"


def test_evaluate_contract(tmp_path):
    client = _make_client(tmp_path)
    initial_params = client.get_parameters(config={})
    loss, n_test, metrics = client.evaluate(initial_params, config={})

    assert isinstance(loss, float)
    assert n_test == 20
    for key in ("threat_macro_f1",
                "escalation_mae", "escalation_auroc", "escalation_spearman",
                "total_loss", "threat_loss", "escalation_loss",
                "threat_class_0_precision", "threat_class_1_recall"):
        assert key in metrics, f"missing eval metric {key}"


def test_round_count_increments(tmp_path):
    client = _make_client(tmp_path)
    initial = client.get_parameters(config={})
    assert client.round_count == 0
    client.fit(initial, config={})
    assert client.round_count == 1
    client.fit(client.get_parameters(config={}), config={})
    assert client.round_count == 2


def test_logs_get_written(tmp_path):
    client = _make_client(tmp_path)
    initial = client.get_parameters(config={})
    client.fit(initial, config={})
    client.evaluate(initial, config={})

    train_log = (tmp_path / "train.log").read_text()
    eval_log = (tmp_path / "eval.log").read_text()
    assert "Round: 1" in train_log
    assert "threat_macro_f1" in eval_log


# ───────────────────────────────────────────────────────────────────────
#   Issue #3 — empty partition guard
# ───────────────────────────────────────────────────────────────────────

def test_fit_with_empty_partition_returns_zero_weight(tmp_path):
    """N=10 geographic partitioning can land a region empty after the
    train/val split. ``fit()`` must return ``num_examples=0`` so FedAvg
    drops the client from the round rather than crashing on the empty
    tf.data.Dataset.shuffle()."""
    client = _make_client(tmp_path)
    initial = client.get_parameters(config={})

    # Override training data with empty arrays (still correct shape).
    client.x_train = np.empty((0, 6), dtype=np.float32)
    client.y_train_class = np.empty((0,), dtype=np.int64)
    client.y_train_esc = np.empty((0,), dtype=np.float32)

    new_params, n_train, metrics = client.fit(initial, config={})

    assert n_train == 0
    # Weights returned unchanged so FedAvg sees a no-op contribution.
    for n, i in zip(new_params, initial):
        np.testing.assert_array_equal(n, i)
    assert metrics["skipped_empty_partition"] == 1.0


def test_evaluate_with_empty_test_returns_nan(tmp_path):
    """Symmetric guard — empty test set should not crash evaluation."""
    client = _make_client(tmp_path)
    initial = client.get_parameters(config={})

    client.x_test = np.empty((0, 6), dtype=np.float32)
    client.y_test_class = np.empty((0,), dtype=np.int64)
    client.y_test_esc = np.empty((0,), dtype=np.float32)

    loss, n_test, metrics = client.evaluate(initial, config={})
    assert n_test == 0
    assert metrics["skipped_empty_partition"] == 1.0


# ───────────────────────────────────────────────────────────────────────
#   Issue #6 — fit() captures config["mu"] for Phase D FedProx
# ───────────────────────────────────────────────────────────────────────

def test_fit_captures_proximal_mu_from_config(tmp_path):
    """Phase D's FedProx broadcasts ``mu`` via the per-round config.
    Phase C documents the hook by capturing the value on the client;
    the actual loss-augmentation is deferred to Phase D."""
    client = _make_client(tmp_path)
    initial = client.get_parameters(config={})
    client.fit(initial, config={"mu": 0.01})
    assert client.proximal_mu == pytest.approx(0.01)


def test_fit_proximal_mu_defaults_to_zero(tmp_path):
    client = _make_client(tmp_path)
    initial = client.get_parameters(config={})
    client.fit(initial, config={})
    assert client.proximal_mu == 0.0


# ───────────────────────────────────────────────────────────────────────
#   Issue #11 — print_summary flag (default off for N-client simulation)
# ───────────────────────────────────────────────────────────────────────

def test_print_summary_default_off(tmp_path, capsys):
    _make_client(tmp_path)  # construction triggers compile, but no summary
    captured = capsys.readouterr()
    assert "Total params" not in captured.out


def test_print_summary_flag_on(tmp_path, capsys):
    _make_client(tmp_path, print_summary_enabled=True)
    captured = capsys.readouterr()
    assert "Total params" in captured.out


# ───────────────────────────────────────────────────────────────────────
#   Phase D review #5 — full FedProx pipeline through FlFusionMLPClient
# ───────────────────────────────────────────────────────────────────────

def test_fl_client_with_fedprox_model_completes_fit(tmp_path):
    """End-to-end smoke for FedProx: build a FedProxFusionMLPModel-backed
    FL client, call fit() with the FULL parameters list (as Flower does
    in production), verify the proximal pipeline doesn't crash on
    shape mismatch (Phase D review #1) and the model trains."""
    from Config.modelStructures.FusionMLP.multiTaskMLPStruct import (
        build_fedprox_fusion_mlp,
    )

    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 64, 16, 16
    X_train = rng.standard_normal((n_train, 5)).astype("float32")
    X_val = rng.standard_normal((n_val, 5)).astype("float32")
    X_test = rng.standard_normal((n_test, 5)).astype("float32")
    y_train = (rng.integers(0, 3, n_train).astype("int64"),
                rng.random(n_train).astype("float32"))
    y_val = (rng.integers(0, 3, n_val).astype("int64"),
              rng.random(n_val).astype("float32"))
    y_test = (rng.integers(0, 3, n_test).astype("int64"),
               rng.random(n_test).astype("float32"))

    model = build_fedprox_fusion_mlp(input_dim=5, num_classes=3,
                                      l2_alpha=0.0, mu=0.0)
    client = FlFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_val, x_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        BATCH_SIZE=16, epochs=1,
        steps_per_epoch=max(1, n_train // 16),
        learning_rate=1e-3,
        evaluation_log=str(tmp_path / "eval.log"),
        training_log=str(tmp_path / "train.log"),
        shuffle_seed=0,
    )

    # Flower-style: full weights list (matching get_weights()).
    initial_params = client.get_parameters(config={})
    assert len(initial_params) == len(model.get_weights())

    # The fit() call must NOT crash on shape mismatch. Pre-fix this
    # crashed inside train_step zipping anchors against trainable_weights.
    new_params, n_examples, metrics = client.fit(
        initial_params, config={"mu": 0.05},
    )

    assert len(new_params) == len(initial_params)
    assert n_examples == n_train
    assert "loss" in metrics
    # Client's proximal_mu reflects the broadcast.
    assert client.proximal_mu == pytest.approx(0.05)
    # And the model's fedprox_mu attribute (now a tf.Variable) matches.
    assert model.fedprox_mu == pytest.approx(0.05)


def test_fl_client_fit_returns_proximal_contribution_metric(tmp_path):
    """The client's fit() return dict carries ``proximal_contribution``
    through from the model's history. For FedAvg models (plain Model)
    the key is still present with value 0 — consistent schema across
    strategies for the server to aggregate."""
    from Config.modelStructures.FusionMLP.multiTaskMLPStruct import (
        build_fedprox_fusion_mlp,
    )

    rng = np.random.default_rng(2)
    n_train = 32
    X_train = rng.standard_normal((n_train, 5)).astype("float32")
    y_train = (rng.integers(0, 3, n_train).astype("int64"),
                rng.random(n_train).astype("float32"))
    X_val = X_train[:8]
    y_val = (y_train[0][:8], y_train[1][:8])

    # FedProx path — mu > 0, anchor set during fit
    model = build_fedprox_fusion_mlp(input_dim=5, num_classes=3,
                                      l2_alpha=0.0, mu=0.0)
    client = FlFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_val, x_test=X_val,
        y_train=y_train, y_val=y_val, y_test=y_val,
        BATCH_SIZE=16, epochs=1, steps_per_epoch=2, learning_rate=1e-3,
        evaluation_log=str(tmp_path / "eval.log"),
        training_log=str(tmp_path / "train.log"),
    )

    initial_params = client.get_parameters(config={})
    # Bump weights via parameters so set_global_weights anchors at a
    # known point and the post-set_weights trainable weights drift from it.
    bumped = [w + 0.2 for w in initial_params]
    _, _, metrics = client.fit(bumped, config={"mu": 0.5})

    assert "proximal_contribution" in metrics
    assert isinstance(metrics["proximal_contribution"], float)
    # The model was perturbed against its anchor by Keras's optimizer
    # update — proximal contribution should be measurable but bounded.
    assert metrics["proximal_contribution"] >= 0.0


def test_fl_client_fit_returns_zero_proximal_when_model_is_plain(tmp_path):
    """A plain (non-FedProx) FusionMLP model has no proximal_contribution
    in its history — the client should default to 0 so the schema stays
    consistent for the strategy aggregation."""
    client = _make_client(tmp_path)
    initial = client.get_parameters(config={})
    _, _, metrics = client.fit(initial, config={"mu": 0.0})

    assert "proximal_contribution" in metrics
    assert metrics["proximal_contribution"] == 0.0


def test_fl_client_fedprox_anchor_aligns_with_trainable_subset(tmp_path):
    """After fit() with a FedProx model, the anchor variables (filtered
    to trainable indices) should match the trainable subset of the
    broadcast parameters. This is the bug Phase D review #1 caught."""
    from Config.modelStructures.FusionMLP.multiTaskMLPStruct import (
        build_fedprox_fusion_mlp,
    )

    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((32, 5)).astype("float32")
    y_train = (rng.integers(0, 3, 32).astype("int64"),
                rng.random(32).astype("float32"))

    model = build_fedprox_fusion_mlp(input_dim=5, num_classes=3,
                                      l2_alpha=0.0, mu=0.0)
    client = FlFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_train[:8], x_test=X_train[:8],
        y_train=y_train,
        y_val=(y_train[0][:8], y_train[1][:8]),
        y_test=(y_train[0][:8], y_train[1][:8]),
        BATCH_SIZE=16, epochs=1, steps_per_epoch=2, learning_rate=1e-3,
        evaluation_log=str(tmp_path / "eval.log"),
        training_log=str(tmp_path / "train.log"),
    )

    # Construct a "broadcast" — full weights list with each entry
    # bumped by a unique constant so we can identify alignment.
    full_initial = model.get_weights()
    bumped = [w + 0.1 * (i + 1) for i, w in enumerate(full_initial)]

    client.fit(bumped, config={"mu": 0.05})

    # _trainable_indices was populated during set_global_weights.
    trainable_indices = model.__dict__["_trainable_indices"]
    anchors = model.__dict__["_global_weight_vars"]
    assert len(anchors) == len(trainable_indices)

    # Anchor i should equal bumped[trainable_indices[i]].
    for i, ti in enumerate(trainable_indices):
        np.testing.assert_array_equal(anchors[i].numpy(), bumped[ti])
