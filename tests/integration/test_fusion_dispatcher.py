"""Phase A — dispatcher-level integration tests for ``load_commcrime``.

Covers issues 1, 3, 4 from the Phase A review (run_dir reuse, hermes
rejection, audit-in-stats). Exercises the same path the dispatcher
takes from `datasetLoadProcess.py`, just imported directly to avoid
pulling in the legacy module's flwr/tf top-level imports.

The Phase A unit tests cover the individual building blocks; this
suite covers their composition under realistic args namespaces.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    make_synthetic_stub, stub_names_file,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLoadProcess import (
    load_commcrime,
)


def _make_args(tmp_path: Path, *,
               run_dir: str | None = None,
               num_clients: int = 5,
               client_id: int = 0,
               partition_strategy: str = "iid",
               drop_sensitive_features: bool = True,
               mode: str = "legacy") -> SimpleNamespace:
    """Build a SimpleNamespace mirroring the argparse output for COMMCRIME runs."""
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=300, seed=7)
    stub_names_file(tmp_path / "stub.names")
    # The stub .names file sits next to the data file by default; load_commcrime
    # falls back to data_path.parent / RAW_NAMES_FILENAME which expects the
    # canonical filename. Symlink/copy the stub .names to that canonical name.
    canonical_names = csv.with_name("communities_and_crime_unnormalized.names")
    canonical_names.write_text((tmp_path / "stub.names").read_text())

    return SimpleNamespace(
        timestamp="20260510_120000",
        mode=mode,
        commcrime_path=str(csv),
        commcrime_random_seed=42,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        dirichlet_alpha=0.5,
        client_id=client_id,
        global_test_size=0.15,
        drop_sensitive_features=drop_sensitive_features,
        dataset_processing="COMMCRIME",
        run_dir=run_dir,
        # argparse normally sets these; mirror it so the dispatcher's
        # log re-routing has something to operate on.
        evaluationLog="20260510_120000_evaluation.log",
        trainingLog="20260510_120000_training.log",
    )


# ───────────────────────────────────────────────────────────────────────
#   End-to-end: dispatcher returns the standard 6-tuple
# ───────────────────────────────────────────────────────────────────────

def test_load_commcrime_returns_six_tuple_with_y_pairs(tmp_path):
    args = _make_args(tmp_path)
    result = load_commcrime(args)
    assert len(result) == 6
    X_train, X_val, y_train, y_val, X_test, y_test = result
    assert isinstance(y_train, tuple) and len(y_train) == 2
    assert isinstance(y_val, tuple) and len(y_val) == 2
    assert isinstance(y_test, tuple) and len(y_test) == 2
    assert X_train.dtype == np.float32
    assert y_train[0].dtype == np.int64
    assert y_train[1].dtype == np.float32


def test_load_commcrime_writes_run_artifacts(tmp_path):
    args = _make_args(tmp_path, run_dir=str(tmp_path / "myrun"))
    load_commcrime(args)
    run_dir = Path(args.run_dir)
    assert (run_dir / "env_pip_freeze.txt").exists()
    assert (run_dir / "partition_stats.json").exists()
    assert (run_dir / "scaler.joblib").exists()
    assert (run_dir / "partitions" / "global_test.pkl").exists()
    for cid in range(5):
        assert (run_dir / "partitions" / f"client_{cid}.pkl").exists()


# ───────────────────────────────────────────────────────────────────────
#   Issue #1 — --run_dir honored across calls
# ───────────────────────────────────────────────────────────────────────

def test_explicit_run_dir_is_honored(tmp_path):
    target = tmp_path / "explicit_run"
    args = _make_args(tmp_path, run_dir=str(target))
    load_commcrime(args)
    assert target.exists()
    assert (target / "partitions" / "global_test.pkl").exists()


def test_two_clients_share_run_dir_share_global_test(tmp_path):
    """Phase C will spawn N processes that must all see the same global test.

    Simulates two clients invoking ``load_commcrime`` with the same
    ``--run_dir``; their ``y_test`` arrays must be identical.
    """
    shared = tmp_path / "shared_run"
    args_a = _make_args(tmp_path, run_dir=str(shared), client_id=0)
    args_b = _make_args(tmp_path, run_dir=str(shared), client_id=1)

    *_, y_test_a = load_commcrime(args_a)
    *_, y_test_b = load_commcrime(args_b)

    np.testing.assert_array_equal(y_test_a[0], y_test_b[0])
    np.testing.assert_array_equal(y_test_a[1], y_test_b[1])


def test_no_run_dir_creates_fresh_timestamped_dir(tmp_path, monkeypatch):
    """When --run_dir is None, fall back to results/commcrime_run_<timestamp>/."""
    monkeypatch.chdir(tmp_path)  # results/ is created under CWD
    args = _make_args(tmp_path, run_dir=None)
    load_commcrime(args)
    auto_dir = Path(args.run_dir)
    assert auto_dir.exists()
    assert auto_dir.name.startswith("commcrime_run_")
    assert auto_dir.parent.name == "results"


# ───────────────────────────────────────────────────────────────────────
#   Issue #3 — --mode hermes is rejected
# ───────────────────────────────────────────────────────────────────────

def test_hermes_mode_rejected(tmp_path):
    args = _make_args(tmp_path, mode="hermes")
    with pytest.raises(SystemExit, match="does not support --mode hermes"):
        load_commcrime(args)


def test_legacy_mode_accepted(tmp_path):
    args = _make_args(tmp_path, mode="legacy")
    # No exception expected; smoke test only.
    load_commcrime(args)


# ───────────────────────────────────────────────────────────────────────
#   Issue #4 — audit() output lands in partition_stats.json
# ───────────────────────────────────────────────────────────────────────

def test_audit_info_in_partition_stats(tmp_path):
    args = _make_args(tmp_path, run_dir=str(tmp_path / "run"))
    load_commcrime(args)
    stats = json.loads(Path(args.run_dir, "partition_stats.json").read_text())
    assert "audit" in stats
    audit = stats["audit"]
    # Pre-cleaning audit: n_rows matches the synthetic stub, n_cols
    # matches the stub schema, state_value_counts is non-empty.
    assert audit["n_rows"] == 300
    assert audit["n_cols"] > 0
    assert "state_value_counts" in audit
    assert len(audit["state_value_counts"]) > 0


# ───────────────────────────────────────────────────────────────────────
#   Issue #8 — dropped sensitive columns recorded per-run
# ───────────────────────────────────────────────────────────────────────

def test_dispatcher_records_dropped_sensitive_columns_default(tmp_path):
    """Default --drop_sensitive_features=True → at least the columns the
    synthetic stub contains from SENSITIVE_COLUMNS are recorded."""
    args = _make_args(tmp_path, run_dir=str(tmp_path / "run"),
                       drop_sensitive_features=True)
    load_commcrime(args)
    stats = json.loads(Path(args.run_dir, "partition_stats.json").read_text())
    dropped = stats["dropped_sensitive_columns"]
    # Stub includes racepctblack, racePctWhite, medIncome, pctWInvInc;
    # those should all be in the dropped list.
    assert "racepctblack" in dropped
    assert "medIncome" in dropped


def test_dispatcher_records_empty_dropped_when_keep_sensitive(tmp_path):
    """--no-drop_sensitive_features → empty dropped list."""
    args = _make_args(tmp_path, run_dir=str(tmp_path / "run"),
                       drop_sensitive_features=False)
    load_commcrime(args)
    stats = json.loads(Path(args.run_dir, "partition_stats.json").read_text())
    assert stats["dropped_sensitive_columns"] == []


# ───────────────────────────────────────────────────────────────────────
#   Issue #2 — re-run conflict detection at the dispatcher level
# ───────────────────────────────────────────────────────────────────────

def test_dispatcher_rerun_idempotent_on_matching_inputs(tmp_path):
    """Two dispatcher calls with the same args + same --run_dir succeed."""
    rd = str(tmp_path / "shared")
    load_commcrime(_make_args(tmp_path, run_dir=rd))
    load_commcrime(_make_args(tmp_path, run_dir=rd))


def test_dispatcher_rerun_raises_on_strategy_conflict(tmp_path):
    rd = str(tmp_path / "shared")
    load_commcrime(_make_args(tmp_path, run_dir=rd, partition_strategy="iid"))
    with pytest.raises(ValueError, match="conflict"):
        load_commcrime(_make_args(tmp_path, run_dir=rd,
                                   partition_strategy="geographic"))
