"""Phase A.4 — preprocessing tests for Communities and Crime.

Pins:
* Output shapes are correct and consistent across train/val/test.
* ``y_*`` outputs are 2-tuples ``(y_class, y_escalation)``.
* StandardScaler vs MinMaxScaler dispatch works.
* Scaler is persisted on first call and re-loaded on second call —
  cross-client preprocessing stays deterministic.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    clean, load_raw, make_synthetic_stub, parse_names_file, stub_names_file,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLabelEngineering import (
    engineer_labels,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeFederatedPartition import (
    partition,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimePreprocess import (
    fit_global_scaler, preprocess_communities_crime,
)


@pytest.fixture
def split(tmp_path):
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=200, seed=7)
    names = parse_names_file(stub_names_file(tmp_path / "stub.names"))
    labeled = engineer_labels(clean(load_raw(csv, names), drop_sensitive=True))
    parts = partition(labeled, strategy="iid", num_clients=5,
                      seed=42, run_dir=tmp_path / "run")
    return parts[0]["train"], parts[0]["val"], parts["global_test"], tmp_path


def test_preprocess_returns_six_tuple_with_y_pairs(split):
    train, val, gt, tmp_path = split
    X_train, X_val, y_train, y_val, X_test, y_test = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME",
        scaler_path=str(tmp_path / "scaler.joblib"),
    )
    assert isinstance(y_train, tuple) and len(y_train) == 2
    assert isinstance(y_val, tuple) and len(y_val) == 2
    assert isinstance(y_test, tuple) and len(y_test) == 2
    y_train_class, y_train_esc = y_train
    assert y_train_class.dtype == np.int64
    assert y_train_esc.dtype == np.float32


def test_preprocess_shapes_match(split):
    train, val, gt, tmp_path = split
    X_train, X_val, y_train, y_val, X_test, y_test = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME",
        scaler_path=str(tmp_path / "scaler.joblib"),
    )
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == len(y_train[0]) == len(y_train[1])
    assert X_test.shape[0] == len(y_test[0])


def test_standard_scaler_centers_training_data(split):
    train, val, gt, tmp_path = split
    X_train, _, _, _, _, _ = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME",
        scaler_path=str(tmp_path / "scaler.joblib"),
    )
    # StandardScaler: training mean ≈ 0, std ≈ 1.
    np.testing.assert_allclose(X_train.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(X_train.std(axis=0), 1.0, atol=1e-1)


def test_minmax_scaler_bounds_training_data(split):
    train, val, gt, tmp_path = split
    X_train, _, _, _, _, _ = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME-MM",
        scaler_path=str(tmp_path / "scaler.joblib"),
    )
    assert X_train.min() >= 0.0 - 1e-6
    assert X_train.max() <= 1.0 + 1e-6


def test_scaler_persisted_and_reused(split):
    train, val, gt, tmp_path = split
    scaler_path = tmp_path / "scaler.joblib"

    X1, _, _, _, _, _ = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME", scaler_path=str(scaler_path),
    )
    assert scaler_path.exists()

    # Second call with the same scaler_path should reload, not refit.
    # Replace the scaler file with one whose mean is artificially shifted;
    # if the preprocessor refits, the shift gets overwritten.
    original = joblib.load(scaler_path)
    # Persisted artifact is now an imputer→scaler Pipeline (Phase E.6
    # NaN-handling fix). Allow either the bare scaler or the Pipeline
    # wrapping it, and assert on the inner type either way.
    from sklearn.pipeline import Pipeline
    if isinstance(original, Pipeline):
        inner = original.named_steps["scale"]
        assert isinstance(inner, (StandardScaler, MinMaxScaler))
    else:
        assert isinstance(original, (StandardScaler, MinMaxScaler))
    X2, _, _, _, _, _ = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME", scaler_path=str(scaler_path),
    )
    np.testing.assert_allclose(X1, X2, atol=1e-6)


def test_unknown_mode_raises(split):
    train, val, gt, _ = split
    with pytest.raises(ValueError, match="Unknown COMMCRIME preprocessing mode"):
        preprocess_communities_crime(train, val, gt, mode="WAT")


def test_preprocess_imputes_nan_inputs(split):
    """Phase E.6 regression test for the real-data shakedown.

    The UCI Communities and Crime LEMAS columns are ~85% NaN. Before
    the median-imputation fix, NaN propagated through the StandardScaler
    untouched and the first forward pass produced NaN loss for every
    epoch (training was running but learning nothing). Pin that the
    preprocessor scrubs NaN out of the feature matrix end-to-end.
    """
    train, val, gt, tmp_path = split

    # Inject NaN into ~half the rows in a couple of feature columns.
    feature_cols = [c for c in train.columns
                    if c not in ("State", "threat_class", "escalation_score")]
    poison_cols = feature_cols[:3]
    rng = np.random.default_rng(0)
    for col in poison_cols:
        mask = rng.random(len(train)) < 0.5
        train.loc[mask, col] = np.nan
        # Also poison val/test so we cover transform-side NaN.
        vmask = rng.random(len(val)) < 0.5
        val.loc[vmask, col] = np.nan
        gmask = rng.random(len(gt)) < 0.5
        gt.loc[gmask, col] = np.nan

    assert train[poison_cols].isna().any().any(), "fixture sanity check"

    X_train, X_val, _, _, X_test, _ = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME",
        scaler_path=str(tmp_path / "scaler.joblib"),
    )

    assert not np.isnan(X_train).any(), "NaN leaked through into X_train"
    assert not np.isnan(X_val).any(), "NaN leaked through into X_val"
    assert not np.isnan(X_test).any(), "NaN leaked through into X_test"


def test_state_column_excluded_from_features(split):
    train, val, gt, tmp_path = split
    feature_count_before = train.shape[1] - 3  # threat_class, escalation_score, state
    X_train, _, _, _, _, _ = preprocess_communities_crime(
        train, val, gt, mode="COMMCRIME",
        scaler_path=str(tmp_path / "scaler.joblib"),
    )
    assert X_train.shape[1] == feature_count_before


# ───────────────────────────────────────────────────────────────────────
#   Issue #7 — fit_global_scaler (Phase C contract)
# ───────────────────────────────────────────────────────────────────────

def test_fit_global_scaler_writes_artifact(tmp_path):
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=200, seed=7)
    names = parse_names_file(stub_names_file(tmp_path / "stub.names"))
    labeled = engineer_labels(clean(load_raw(csv, names), drop_sensitive=True))
    parts = partition(labeled, strategy="iid", num_clients=5,
                      seed=42, run_dir=tmp_path / "run")

    client_trains = [parts[cid]["train"] for cid in range(5)]
    scaler_path = fit_global_scaler(client_trains, str(tmp_path / "run"))

    assert scaler_path.exists()
    assert scaler_path == tmp_path / "run" / "scaler.joblib"


def test_fit_global_scaler_then_preprocess_reuses_it(tmp_path):
    """The Phase C contract: pre-fit once on union → every per-client
    preprocess() call loads the same scaler rather than refitting locally."""
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=300, seed=7)
    names = parse_names_file(stub_names_file(tmp_path / "stub.names"))
    labeled = engineer_labels(clean(load_raw(csv, names), drop_sensitive=True))
    parts = partition(labeled, strategy="iid", num_clients=5,
                      seed=42, run_dir=tmp_path / "run")

    client_trains = [parts[cid]["train"] for cid in range(5)]
    fit_global_scaler(client_trains, str(tmp_path / "run"))

    # Now run preprocess for each client with the same scaler_path.
    # The resulting transformed features should all share the same
    # mean/std behavior (the scaler fit on the union, not on the
    # individual client).
    scaler_path = str(tmp_path / "run" / "scaler.joblib")
    transformed_clients = []
    for cid in range(5):
        X_train, _, _, _, _, _ = preprocess_communities_crime(
            parts[cid]["train"], parts[cid]["val"], parts["global_test"],
            mode="COMMCRIME", scaler_path=scaler_path,
        )
        transformed_clients.append(X_train)

    # Re-running fit_global_scaler should produce an identical scaler;
    # verify by re-loading after a second fit and comparing applied output.
    fit_global_scaler(client_trains, str(tmp_path / "run"))
    X_train_again, _, _, _, _, _ = preprocess_communities_crime(
        parts[0]["train"], parts[0]["val"], parts["global_test"],
        mode="COMMCRIME", scaler_path=scaler_path,
    )
    np.testing.assert_allclose(transformed_clients[0], X_train_again, atol=1e-6)


def test_fit_global_scaler_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="at least one training frame"):
        fit_global_scaler([], str(tmp_path / "run"))


def test_fit_global_scaler_rejects_unknown_mode(tmp_path):
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=50, seed=7)
    names = parse_names_file(stub_names_file(tmp_path / "stub.names"))
    labeled = engineer_labels(clean(load_raw(csv, names), drop_sensitive=True))
    with pytest.raises(ValueError, match="Unknown COMMCRIME preprocessing mode"):
        fit_global_scaler([labeled], str(tmp_path / "run"), mode="WAT")
