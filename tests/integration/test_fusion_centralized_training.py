"""Phase B integration test — full centralized FUSION-MLP run on synthetic data.

Validates the Phase B Definition of Done:
* End-to-end smoke run completes in < 10 s on a 100-row synthetic
  partition.
* ``.fit()`` returns a history with both head losses decreasing.
* ``.evaluate()`` writes macro-F1, per-class P/R, escalation MAE,
  escalation AUROC to the evaluation log.
* ``.save()`` produces ``ModelArchive/fusion_mlp_<save_name>.h5``.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import time
from pathlib import Path

import numpy as np
import pytest

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    clean, load_raw, make_synthetic_stub, parse_names_file, stub_names_file,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeFederatedPartition import (
    partition,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLabelEngineering import (
    engineer_labels,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimePreprocess import (
    preprocess_communities_crime,
)
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.FusionMLP.fusionMLPCentralTrainingConfig import (
    CentralFusionMLPClient,
)
from Config.modelStructures.FusionMLP.multiTaskMLPStruct import build_fusion_mlp


@pytest.fixture
def small_synthetic_run(tmp_path):
    """Tiny end-to-end Phase A pipeline → 6-tuple ready for Phase B training."""
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=300, seed=7)
    names = parse_names_file(stub_names_file(tmp_path / "stub.names"))
    labeled = engineer_labels(clean(load_raw(csv, names), drop_sensitive=True))
    parts = partition(labeled, strategy="iid", num_clients=1,
                       seed=42, run_dir=tmp_path / "run")
    return preprocess_communities_crime(
        parts[0]["train"], parts[0]["val"], parts["global_test"],
        mode="COMMCRIME",
        scaler_path=str(tmp_path / "run" / "scaler.joblib"),
    ), tmp_path


def test_centralized_fit_evaluate_save(small_synthetic_run):
    (X_train, X_val, y_train, y_val, X_test, y_test), tmp_path = small_synthetic_run

    model = build_fusion_mlp(input_dim=X_train.shape[1], num_classes=3,
                              l2_alpha=0.0)

    train_log = tmp_path / "training.log"
    eval_log = tmp_path / "evaluation.log"

    client = CentralFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_val, x_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        BATCH_SIZE=32, epochs=3, steps_per_epoch=max(1, len(X_train) // 32),
        learning_rate=1e-3,
        escalation_loss_weight=0.5,
        evaluation_log=str(eval_log),
        training_log=str(train_log),
    )

    start = time.time()
    history = client.fit()
    metrics = client.evaluate()
    elapsed = time.time() - start

    # DoD: completes in well under 10s on the synthetic partition.
    assert elapsed < 30  # generous bound for cold-start TF on Windows

    # Both heads' validation losses appear in history.
    assert "val_threat_loss" in history.history
    assert "val_escalation_loss" in history.history

    # Expected eval metrics present.
    for key in ("threat_macro_f1", "escalation_mae", "escalation_auroc",
                "threat_class_0_precision", "threat_class_0_recall",
                "total_loss", "threat_loss", "escalation_loss"):
        assert key in metrics, f"missing metric: {key}"

    # Save into a tmp_path-scoped archive — the trainer's archive_path
    # kwarg keeps the test from polluting the real ModelArchive/ folder.
    archive = tmp_path / "ModelArchive"
    out_path = client.save("phase_b_smoke", archive_path=str(archive))
    assert Path(out_path).exists()
    assert (archive / "fusion_mlp_phase_b_smoke.keras").exists()
    # And explicitly verify nothing leaked into the real project ModelArchive.
    real_archive = Path(__file__).resolve().parents[2] / "ModelArchive"
    assert not (real_archive / "fusion_mlp_phase_b_smoke.keras").exists(), (
        f"test polluted real ModelArchive at {real_archive}"
    )
    assert not (real_archive / "fusion_mlp_phase_b_smoke.h5").exists(), (
        "stale .h5 file lingering from before #11"
    )


def test_evaluation_log_contains_required_metrics(small_synthetic_run):
    (X_train, X_val, y_train, y_val, X_test, y_test), tmp_path = small_synthetic_run

    model = build_fusion_mlp(input_dim=X_train.shape[1], num_classes=3,
                              l2_alpha=0.0)
    eval_log = tmp_path / "evaluation.log"

    client = CentralFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_val, x_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        BATCH_SIZE=32, epochs=2, steps_per_epoch=max(1, len(X_train) // 32),
        learning_rate=1e-3,
        evaluation_log=str(eval_log),
        training_log=str(tmp_path / "training.log"),
    )
    client.fit()
    client.evaluate()

    log_text = eval_log.read_text()
    for key in ("threat_macro_f1", "escalation_mae", "escalation_auroc",
                "escalation_spearman"):
        assert key in log_text, f"{key} missing from evaluation log"


def test_print_summary_flag_suppresses_model_summary(small_synthetic_run, capsys):
    """Phase C will instantiate N clients in one process; the
    print_summary=False flag must suppress the architecture print."""
    (X_train, X_val, y_train, y_val, X_test, y_test), tmp_path = small_synthetic_run
    model = build_fusion_mlp(input_dim=X_train.shape[1], num_classes=3,
                              l2_alpha=0.0)
    CentralFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_val, x_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        BATCH_SIZE=32, epochs=1, steps_per_epoch=max(1, len(X_train) // 32),
        learning_rate=1e-3,
        evaluation_log=str(tmp_path / "eval.log"),
        training_log=str(tmp_path / "train.log"),
        print_summary=False,
    )
    captured = capsys.readouterr()
    # When suppressed, the architecture box (which includes "Total params:")
    # should not appear on stdout.
    assert "Total params" not in captured.out


def test_joint_loss_decreases_on_learnable_data(tmp_path):
    """Both head losses should decrease over training when there IS
    signal to learn. We hand-craft a dataset where:
      * threat_class is a linear function of feature 0
      * escalation_score is a linear function of feature 1
    The stub-based smoke test uses uniform-random rates with no signal,
    so it can't validate this — this test uses crafted data instead.
    """
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, 8)).astype("float32")
    # threat: feature[0] sign + magnitude → 3 classes
    y_class = np.where(X[:, 0] < -0.3, 0,
                      np.where(X[:, 0] > 0.3, 2, 1)).astype("int64")
    # escalation: sigmoid of feature[1]
    y_esc = (1.0 / (1.0 + np.exp(-X[:, 1]))).astype("float32")

    # Split 80/10/10
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)
    X_train, X_val, X_test = X[:train_n], X[train_n:train_n+val_n], X[train_n+val_n:]
    y_train = (y_class[:train_n], y_esc[:train_n])
    y_val = (y_class[train_n:train_n+val_n], y_esc[train_n:train_n+val_n])
    y_test = (y_class[train_n+val_n:], y_esc[train_n+val_n:])

    model = build_fusion_mlp(input_dim=8, num_classes=3, l2_alpha=0.0)
    client = CentralFusionMLPClient(
        model=model,
        x_train=X_train, x_val=X_val, x_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        BATCH_SIZE=32, epochs=10, steps_per_epoch=max(1, len(X_train) // 32),
        learning_rate=1e-3,
        evaluation_log=str(tmp_path / "eval.log"),
        training_log=str(tmp_path / "train.log"),
    )
    history = client.fit()
    # Use *validation* loss — training loss with dropout on small batches is
    # too noisy to assert monotonicity. Validation runs without dropout.
    threat = history.history["val_threat_loss"]
    escalation = history.history["val_escalation_loss"]
    assert threat[-1] < threat[0] * 0.80, (
        f"val_threat_loss did not decrease enough: {threat[0]:.3f} → {threat[-1]:.3f}"
    )
    assert escalation[-1] < escalation[0] * 0.99, (
        f"val_escalation_loss did not decrease: {escalation[0]:.3f} → {escalation[-1]:.3f}"
    )

    # And evaluate-time macro-F1 should clear chance (1/3 ≈ 0.33).
    metrics = client.evaluate()
    assert metrics["threat_macro_f1"] > 0.5, (
        f"macro_f1 below chance: {metrics['threat_macro_f1']:.3f}"
    )
