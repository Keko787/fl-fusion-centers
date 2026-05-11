"""Feature scaling + (X, y) splitting for the COMMCRIME data path.

Phase A.4 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

Kept in its own module (rather than appended to the legacy
``datasetPreprocess.py``) so the Phase A tests can import it without
pulling in ``flwr`` / ``tensorflow`` — both of which the legacy module
imports unconditionally at top level. ``datasetPreprocess.py`` re-exports
``preprocess_communities_crime`` so the existing dispatcher import path
is preserved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess_communities_crime(client_train_df: pd.DataFrame,
                                  client_val_df: pd.DataFrame,
                                  global_test_df: pd.DataFrame,
                                  mode: str = "COMMCRIME",
                                  scaler_path: str | None = None,
                                  state_col: str = "state",
                                  label_class_col: str = "threat_class",
                                  label_escalation_col: str = "escalation_score"):
    """Scale features + split labels for the fusion-centers data path.

    Returns the same 6-tuple shape every existing trainer consumes:
    ``(X_train, X_val, y_train, y_val, X_test, y_test)``. The ``y_*``
    entries are 2-tuples ``(y_class, y_escalation)`` instead of single
    arrays — only the FUSION-MLP trainer unpacks them; every other
    trainer branch in ``modelCentralTrainingConfigLoad`` /
    ``modelFederatedTrainingConfigLoad`` ignores this slot, so the
    existing IoT/GAN paths are unaffected.

    Scaler persistence (outline §6.9): if ``scaler_path`` is given and
    the file exists, the fitted scaler is loaded and re-applied —
    making cross-run + cross-client preprocessing deterministic. If
    the file does not exist, a new scaler is fit on the client's
    training partition and saved.
    """
    def _split_xy(df: pd.DataFrame):
        feature_df = df.drop(columns=[c for c in
                                       (state_col, label_class_col, label_escalation_col)
                                       if c in df.columns])
        y_class = df[label_class_col].to_numpy().astype("int64")
        y_esc = df[label_escalation_col].to_numpy().astype("float32")
        return feature_df, y_class, y_esc

    X_train_df, y_train_class, y_train_esc = _split_xy(client_train_df)
    X_val_df, y_val_class, y_val_esc = _split_xy(client_val_df)
    X_test_df, y_test_class, y_test_esc = _split_xy(global_test_df)

    if mode == "COMMCRIME":
        scaler_cls = StandardScaler
    elif mode == "COMMCRIME-MM":
        scaler_cls = MinMaxScaler
    else:
        raise ValueError(f"Unknown COMMCRIME preprocessing mode: {mode!r}")

    if scaler_path is not None and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = scaler_cls().fit(X_train_df.to_numpy())
        if scaler_path is not None:
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)

    X_train = scaler.transform(X_train_df.to_numpy()).astype("float32")
    X_val = scaler.transform(X_val_df.to_numpy()).astype("float32")
    X_test = scaler.transform(X_test_df.to_numpy()).astype("float32")

    print(f"\n=== COMMCRIME preprocess (mode={mode}) ===")
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    print(f"  class dist (train): {np.bincount(y_train_class, minlength=3).tolist()}")

    return (X_train, X_val,
            (y_train_class, y_train_esc),
            (y_val_class, y_val_esc),
            X_test,
            (y_test_class, y_test_esc))


def _scaler_for_mode(mode: str):
    if mode == "COMMCRIME":
        return StandardScaler()
    if mode == "COMMCRIME-MM":
        return MinMaxScaler()
    raise ValueError(f"Unknown COMMCRIME preprocessing mode: {mode!r}")


def fit_global_scaler(client_train_dfs: Iterable[pd.DataFrame],
                       run_dir: str,
                       mode: str = "COMMCRIME",
                       state_col: str = "state",
                       label_class_col: str = "threat_class",
                       label_escalation_col: str = "escalation_score") -> Path:
    """Fit one scaler on the union of all client training partitions.

    Outline §6.5 calls for global feature standardization across the
    federation: fit the scaler once on every client's training data,
    persist it, then have each client's preprocessing call load and
    apply it (rather than re-fitting locally). This helper is the
    "fit-once" half; the per-client call site is
    :func:`preprocess_communities_crime` with ``scaler_path`` pointed
    at the artifact this function writes.

    The simulation runner (Phase C) is expected to:
      1. Load all client training partitions via ``load_partition``.
      2. Call ``fit_global_scaler`` once on the union, passing the
         shared ``run_dir``.
      3. Launch the per-client Flower clients; their preprocessing
         steps will all load the same scaler from
         ``<run_dir>/scaler.joblib``.

    Returns the path to the persisted scaler.
    """
    frames = list(client_train_dfs)
    if not frames:
        raise ValueError("fit_global_scaler requires at least one training frame")

    union = pd.concat(frames, ignore_index=True)
    drop_cols = [c for c in (state_col, label_class_col, label_escalation_col)
                 if c in union.columns]
    feature_df = union.drop(columns=drop_cols)

    scaler = _scaler_for_mode(mode).fit(feature_df.to_numpy())

    scaler_path = Path(run_dir) / "scaler.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"=== fit_global_scaler: {type(scaler).__name__} fit on "
          f"{len(feature_df)} rows × {feature_df.shape[1]} features → {scaler_path}")
    return scaler_path
