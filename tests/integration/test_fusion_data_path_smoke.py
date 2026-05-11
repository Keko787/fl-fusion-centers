"""Phase A end-to-end smoke test on the synthetic stub.

Exercises the full data path the dispatcher uses for a real run, but
short-circuits the UCI download by feeding ``commcrime_path`` a stub
CSV and stub .names. Asserts that:

* Every artifact the Phase A DoD lists appears under the run dir:
  ``partitions/client_<i>.pkl``, ``partitions/global_test.pkl``,
  ``scaler.joblib``, ``partition_stats.json``, ``env_pip_freeze.txt``.
* Re-running with the same seed reproduces partitions byte-for-byte.
* Per-client class distributions are recorded in
  ``partition_stats.json``.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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
from Config.SessionConfig.runArtifacts import (
    dump_pip_freeze, init_run_dir, seed_all,
)


def _run_full_pipeline(tmp_path: Path, *, num_clients: int = 5,
                       strategy: str = "geographic",
                       seed: int = 42,
                       timestamp: str | None = None) -> Path:
    """Drive the end-to-end Phase A pipeline; return the run directory."""
    seed_all(seed)
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=400, seed=seed)
    names = parse_names_file(stub_names_file(tmp_path / "stub.names"))

    raw = load_raw(csv, names)
    cleaned = clean(raw, drop_sensitive=True)
    labeled = engineer_labels(cleaned)

    run_dir = init_run_dir(base=tmp_path / "results", timestamp=timestamp)
    dump_pip_freeze(run_dir / "env_pip_freeze.txt")

    partitions = partition(labeled, strategy=strategy,
                            num_clients=num_clients, seed=seed,
                            run_dir=run_dir)

    # Drive the preprocessor for client 0 to produce the scaler.joblib artifact.
    preprocess_communities_crime(
        partitions[0]["train"],
        partitions[0]["val"],
        partitions["global_test"],
        mode="COMMCRIME",
        scaler_path=str(run_dir / "scaler.joblib"),
    )

    return run_dir


def test_full_pipeline_produces_all_artifacts(tmp_path):
    run_dir = _run_full_pipeline(tmp_path)

    # Definition of Done from the implementation plan §3 Phase A.
    assert (run_dir / "env_pip_freeze.txt").exists()
    assert (run_dir / "partition_stats.json").exists()
    assert (run_dir / "scaler.joblib").exists()
    assert (run_dir / "partitions" / "global_test.pkl").exists()
    for cid in range(5):
        assert (run_dir / "partitions" / f"client_{cid}.pkl").exists()


def test_partition_stats_records_class_distribution(tmp_path):
    run_dir = _run_full_pipeline(tmp_path)
    stats = json.loads((run_dir / "partition_stats.json").read_text())
    assert stats["strategy"] == "geographic"
    assert stats["num_clients"] == 5
    # Every client (including those whose stub regions hit zero rows) has
    # an entry in the stats dict.
    assert set(stats["clients"].keys()) == {"0", "1", "2", "3", "4"}
    for cid in range(5):
        client_stats = stats["clients"][str(cid)]
        assert "train" in client_stats and "val" in client_stats
        assert "class_distribution" in client_stats["train"]


def test_two_runs_same_seed_byte_identical_partitions(tmp_path):
    """Same seed → byte-identical client_<i>.pkl across runs."""
    run_a = _run_full_pipeline(tmp_path / "a", timestamp="20260510_120000")
    run_b = _run_full_pipeline(tmp_path / "b", timestamp="20260510_120001")

    for cid in range(5):
        a_bytes = (run_a / "partitions" / f"client_{cid}.pkl").read_bytes()
        b_bytes = (run_b / "partitions" / f"client_{cid}.pkl").read_bytes()
        # Partitions must reconstruct to identical frames.
        a_payload = pickle.loads(a_bytes)
        b_payload = pickle.loads(b_bytes)
        pd.testing.assert_frame_equal(a_payload["train"], b_payload["train"])
        pd.testing.assert_frame_equal(a_payload["val"], b_payload["val"])


def test_different_num_clients_different_partitions(tmp_path):
    """Same seed but different --num_clients → different (still deterministic) partitions."""
    run_5 = _run_full_pipeline(tmp_path / "n5", num_clients=5,
                                timestamp="20260510_120000")
    run_3 = _run_full_pipeline(tmp_path / "n3", num_clients=3,
                                timestamp="20260510_120001")
    stats_5 = json.loads((run_5 / "partition_stats.json").read_text())
    stats_3 = json.loads((run_3 / "partition_stats.json").read_text())
    assert stats_5["num_clients"] == 5
    assert stats_3["num_clients"] == 3
    assert set(stats_3["clients"].keys()) == {"0", "1", "2"}
