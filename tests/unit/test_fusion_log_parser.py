"""Phase E.4 — log parser tests.

Pins the contract between the strategy/trainer log formats and the
plot scripts that consume them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from Analysis.CommunitiesCrime.log_parser import (
    collect_server_logs,
    parse_client_log,
    parse_partition_stats,
    parse_server_log,
)


# ───────────────────────────────────────────────────────────────────────
#   Server log parser
# ───────────────────────────────────────────────────────────────────────

_SERVER_LOG_SAMPLE = """=== Round 1 (5 clients) ===
aggregated_loss: 0.4231
threat_macro_f1: 0.5234
escalation_mae: 0.1812
escalation_auroc: 0.7421
escalation_spearman: 0.4523
fairness_macro_f1_variance: 0.0123
fairness_accuracy_variance: 0.0087
round_seconds: 2.45
parameter_update_wire_bytes: 56248.0
plateau_detected: 0.0

=== Round 2 (5 clients) ===
aggregated_loss: 0.3812
threat_macro_f1: 0.5891
escalation_mae: 0.1654
escalation_auroc: 0.7732
escalation_spearman: 0.4892
fairness_macro_f1_variance: 0.0102
fairness_accuracy_variance: 0.0071
round_seconds: 2.31
parameter_update_wire_bytes: 56248.0
plateau_detected: 0.0
"""


def test_parse_server_log_basic(tmp_path):
    log = tmp_path / "server_evaluation.log"
    log.write_text(_SERVER_LOG_SAMPLE)
    df = parse_server_log(log)
    assert len(df) == 2
    assert list(df["round"]) == [1, 2]
    assert list(df["n_clients"]) == [5, 5]
    assert df["threat_macro_f1"].iloc[1] == pytest.approx(0.5891)
    assert df["round_seconds"].iloc[0] == pytest.approx(2.45)


def test_parse_server_log_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_server_log(tmp_path / "nope.log")


def test_parse_server_log_empty_returns_empty_frame(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("")
    df = parse_server_log(log)
    assert df.empty
    assert "round" in df.columns


def test_parse_server_log_tolerates_non_numeric_values(tmp_path):
    log = tmp_path / "messy.log"
    log.write_text(
        "=== Round 1 (2 clients) ===\n"
        "aggregated_loss: 0.5\n"
        "comment: this is a string value\n"
        "threat_macro_f1: 0.6\n"
    )
    df = parse_server_log(log)
    assert df["threat_macro_f1"].iloc[0] == pytest.approx(0.6)
    # Non-numeric line silently dropped (no `comment` column)
    assert "comment" not in df.columns


# ───────────────────────────────────────────────────────────────────────
#   Client log parser
# ───────────────────────────────────────────────────────────────────────

_CLIENT_LOG_SAMPLE = """Node|0| Round: 1
Evaluation Time Elapsed: 0.42 seconds
threat_macro_f1: 0.55
escalation_mae: 0.18
total_loss: 0.4231

Node|0| Round: 2
Evaluation Time Elapsed: 0.39 seconds
threat_macro_f1: 0.62
escalation_mae: 0.17
total_loss: 0.3812
"""


def test_parse_client_log_basic(tmp_path):
    log = tmp_path / "evaluation.log"
    log.write_text(_CLIENT_LOG_SAMPLE)
    df = parse_client_log(log)
    assert len(df) == 2
    assert list(df["node"]) == [0, 0]
    assert list(df["round"]) == [1, 2]
    assert df["threat_macro_f1"].iloc[1] == pytest.approx(0.62)


# ───────────────────────────────────────────────────────────────────────
#   partition_stats.json
# ───────────────────────────────────────────────────────────────────────

def test_parse_partition_stats_round_trip(tmp_path):
    payload = {
        "strategy": "geographic",
        "num_clients": 5,
        "clients": {"0": {"train": {"n_rows": 100}}},
    }
    f = tmp_path / "partition_stats.json"
    f.write_text(json.dumps(payload))
    loaded = parse_partition_stats(f)
    assert loaded == payload


# ───────────────────────────────────────────────────────────────────────
#   collect_server_logs
# ───────────────────────────────────────────────────────────────────────

def test_collect_server_logs_stacks_with_labels(tmp_path):
    for i, label in enumerate(("a", "b", "c")):
        run_dir = tmp_path / f"run_{label}"
        run_dir.mkdir()
        (run_dir / "server_evaluation.log").write_text(_SERVER_LOG_SAMPLE)

    df = collect_server_logs(
        [tmp_path / f"run_{x}" for x in ("a", "b", "c")],
        labels=["A", "B", "C"],
    )
    assert set(df["label"]) == {"A", "B", "C"}
    # 2 rounds × 3 runs = 6 rows
    assert len(df) == 6


def test_collect_server_logs_default_labels_are_dirname(tmp_path):
    for label in ("alpha", "beta"):
        run_dir = tmp_path / label
        run_dir.mkdir()
        (run_dir / "server_evaluation.log").write_text(_SERVER_LOG_SAMPLE)

    df = collect_server_logs([tmp_path / "alpha", tmp_path / "beta"])
    assert set(df["label"]) == {"alpha", "beta"}


def test_collect_server_logs_length_mismatch_raises(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "server_evaluation.log").write_text(_SERVER_LOG_SAMPLE)
    with pytest.raises(ValueError, match="same length"):
        collect_server_logs([run_dir], labels=["A", "B"])


# ───────────────────────────────────────────────────────────────────────
#   Phase E review #12 — package re-exports
# ───────────────────────────────────────────────────────────────────────

def test_package_reexports_log_parser_helpers():
    """``from Analysis.CommunitiesCrime import parse_server_log`` should
    work without reaching into the submodule (Phase E review #12)."""
    import Analysis.CommunitiesCrime as pkg
    assert hasattr(pkg, "parse_server_log")
    assert hasattr(pkg, "parse_client_log")
    assert hasattr(pkg, "parse_partition_stats")
    assert hasattr(pkg, "collect_server_logs")
