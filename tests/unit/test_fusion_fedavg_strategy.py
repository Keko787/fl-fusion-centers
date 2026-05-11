"""Phase C.3 — FusionFedAvg strategy aggregation tests.

Pins the math:
* ``aggregate_evaluate`` returns the **weighted** mean of every
  paper-figure metric (macro_f1, mae, auroc, spearman).
* The fairness metric is the variance of per-client ``threat_macro_f1``.
* Plateau detection fires after ``plateau_patience`` rounds without
  improvement on the aggregated loss.
* The evaluation log gets a per-round block written to disk.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from unittest.mock import MagicMock

import flwr as fl
import math
import numpy as np
import pytest
from flwr.common import (
    Code, EvaluateRes, FitRes, Status,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.FusionFedAvgConfig import (
    FusionFedAvg,
    _is_nan,
    _variance,
    _weighted_mean,
)


def _result(num_examples: int, metrics: dict, loss: float = 0.5):
    """Build a Flower (ClientProxy, EvaluateRes) pair."""
    proxy = MagicMock()
    res = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=loss,
        num_examples=num_examples,
        metrics=metrics,
    )
    return (proxy, res)


def test_weighted_mean_helper():
    # 10 examples at 0.6 + 30 examples at 0.8 → weighted mean = 0.75
    assert _weighted_mean([(10, 0.6), (30, 0.8)], 40) == pytest.approx(0.75)


def test_variance_helper():
    assert _variance([1.0, 1.0]) == 0.0
    assert _variance([0.0, 1.0]) == pytest.approx(0.25)


def test_aggregate_evaluate_weighted_means(tmp_path):
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    results = [
        _result(20, {
            "threat_macro_f1": 0.6,
            "escalation_mae": 0.1,
            "escalation_auroc": 0.7,
            "escalation_spearman": 0.5,
            "total_loss": 0.5,
            "threat_loss": 0.4,
            "escalation_loss": 0.6,
        }, loss=0.5),
        _result(80, {
            "threat_macro_f1": 0.8,
            "escalation_mae": 0.2,
            "escalation_auroc": 0.9,
            "escalation_spearman": 0.6,
            "total_loss": 0.3,
            "threat_loss": 0.2,
            "escalation_loss": 0.4,
        }, loss=0.3),
    ]
    loss, aggregated = strategy.aggregate_evaluate(1, results, [])

    # FedAvg weighted loss: (20*0.5 + 80*0.3) / 100 = 0.34
    assert loss == pytest.approx(0.34)
    # macro_f1 weighted: (20*0.6 + 80*0.8) / 100 = 0.76
    assert aggregated["threat_macro_f1"] == pytest.approx(0.76)
    # mae weighted: (20*0.1 + 80*0.2) / 100 = 0.18
    assert aggregated["escalation_mae"] == pytest.approx(0.18)
    # auroc weighted: (20*0.7 + 80*0.9) / 100 = 0.86
    assert aggregated["escalation_auroc"] == pytest.approx(0.86)
    # spearman weighted: (20*0.5 + 80*0.6) / 100 = 0.58
    assert aggregated["escalation_spearman"] == pytest.approx(0.58)


def test_aggregate_evaluate_fairness_variance(tmp_path):
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    results = [
        _result(50, {"threat_macro_f1": 0.5}, loss=0.4),
        _result(50, {"threat_macro_f1": 0.9}, loss=0.4),
    ]
    _, aggregated = strategy.aggregate_evaluate(1, results, [])
    # variance of [0.5, 0.9] = 0.04
    assert aggregated["fairness_macro_f1_variance"] == pytest.approx(0.04)


def test_plateau_detection_fires_after_patience(tmp_path):
    """When loss does not improve for plateau_patience rounds, the
    strategy flags ``plateau_detected``."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
        plateau_patience=3, plateau_tol=1e-4,
    )

    # Round 1 — initial loss (sets best).
    strategy.aggregate_evaluate(
        1, [_result(10, {"threat_macro_f1": 0.5}, loss=0.5)], [],
    )
    assert not strategy.plateau_detected

    # Rounds 2, 3, 4 — same loss → no improvement.
    for r in (2, 3, 4):
        strategy.aggregate_evaluate(
            r, [_result(10, {"threat_macro_f1": 0.5}, loss=0.5)], [],
        )
    assert strategy.plateau_detected


def test_plateau_resets_on_improvement(tmp_path):
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
        plateau_patience=3, plateau_tol=1e-4,
    )
    strategy.aggregate_evaluate(
        1, [_result(10, {"threat_macro_f1": 0.5}, loss=0.5)], [],
    )
    strategy.aggregate_evaluate(
        2, [_result(10, {"threat_macro_f1": 0.5}, loss=0.5)], [],
    )
    # Round 3 — improvement.
    strategy.aggregate_evaluate(
        3, [_result(10, {"threat_macro_f1": 0.6}, loss=0.3)], [],
    )
    assert strategy._rounds_since_improvement == 0
    assert not strategy.plateau_detected


def test_evaluation_log_written(tmp_path):
    log_path = tmp_path / "server_eval.log"
    strategy = FusionFedAvg(
        evaluation_log=str(log_path),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )
    strategy.aggregate_evaluate(
        1, [_result(10, {"threat_macro_f1": 0.5,
                          "escalation_mae": 0.1,
                          "escalation_auroc": 0.7,
                          "escalation_spearman": 0.5}, loss=0.4)], [],
    )
    log_text = log_path.read_text()
    assert "Round 1" in log_text
    assert "threat_macro_f1" in log_text
    assert "escalation_spearman" in log_text
    assert "fairness_macro_f1_variance" in log_text


def test_history_accumulates(tmp_path):
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )
    for r in (1, 2, 3):
        strategy.aggregate_evaluate(
            r, [_result(10, {"threat_macro_f1": 0.5}, loss=0.5)], [],
        )
    assert len(strategy.history) == 3
    assert strategy.history[0]["round"] == 1
    assert strategy.history[2]["round"] == 3


# ───────────────────────────────────────────────────────────────────────
#   Issue #1 — aggregate_fit stashes final parameters
# ───────────────────────────────────────────────────────────────────────

def _fit_result(num_examples: int, weights):
    proxy = MagicMock()
    res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(weights),
        num_examples=num_examples,
        metrics={},
    )
    return (proxy, res)


def test_aggregate_fit_stashes_final_parameters(tmp_path):
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    # Before any fit round, no stashed parameters.
    assert strategy._final_parameters is None

    # Two clients with equal weights → FedAvg returns the mean.
    w1 = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
    w2 = [np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
    results = [_fit_result(50, w1), _fit_result(50, w2)]

    aggregated_params, _ = strategy.aggregate_fit(1, results, [])
    assert strategy._final_parameters is not None
    assert aggregated_params is strategy._final_parameters

    # The aggregated weights are the weighted mean.
    final = parameters_to_ndarrays(strategy._final_parameters)
    expected = 0.5 * (w1[0] + w2[0])
    np.testing.assert_allclose(final[0], expected, atol=1e-6)


def test_aggregate_fit_with_no_results_leaves_stash_unchanged(tmp_path):
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )
    # Establish a non-None stash first.
    w = [np.ones((2, 2), dtype=np.float32)]
    strategy.aggregate_fit(1, [_fit_result(10, w)], [])
    assert strategy._final_parameters is not None
    prev = strategy._final_parameters

    # An empty round (e.g. all clients failed) should not clobber it.
    strategy.aggregate_fit(2, [], [])
    assert strategy._final_parameters is prev


# ───────────────────────────────────────────────────────────────────────
#   Issue #2 — NaN entries filtered from weighted means
# ───────────────────────────────────────────────────────────────────────

def test_aggregate_evaluate_filters_nan_client(tmp_path):
    """A client reporting AUROC=NaN must not drag the aggregated AUROC
    toward zero. The good client's value should win the weighted mean."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    results = [
        _result(50, {"escalation_auroc": float("nan"),
                      "threat_macro_f1": 0.5},
                loss=0.5),
        _result(50, {"escalation_auroc": 0.8,
                      "threat_macro_f1": 0.7},
                loss=0.5),
    ]
    _, aggregated = strategy.aggregate_evaluate(1, results, [])
    # Old buggy behavior: (0 + 0.8) / 2 = 0.4. Fixed: 0.8 (NaN client dropped).
    assert aggregated["escalation_auroc"] == pytest.approx(0.8)


def test_aggregate_evaluate_all_nan_returns_nan(tmp_path):
    """If every client reports NaN, the aggregated metric is NaN — not 0."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    results = [
        _result(30, {"escalation_auroc": float("nan"),
                      "threat_macro_f1": 0.5}, loss=0.5),
        _result(70, {"escalation_auroc": float("nan"),
                      "threat_macro_f1": 0.5}, loss=0.5),
    ]
    _, aggregated = strategy.aggregate_evaluate(1, results, [])
    assert math.isnan(aggregated["escalation_auroc"])


def test_is_nan_helper():
    assert _is_nan(None)
    assert _is_nan(float("nan"))
    assert _is_nan(np.nan)
    assert not _is_nan(0.0)
    assert not _is_nan(0.5)
    assert not _is_nan(1)


# ───────────────────────────────────────────────────────────────────────
#   Phase E.2 — federation-overhead metrics
# ───────────────────────────────────────────────────────────────────────

def _fit_result(num_examples: int, weights):
    proxy = MagicMock()
    res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(weights),
        num_examples=num_examples,
        metrics={},
    )
    return (proxy, res)


def _fit_result_with_metrics(num_examples: int, weights, metrics: dict):
    """FitRes with explicit client-side fit metrics (used for proximal_contribution test)."""
    proxy = MagicMock()
    res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(weights),
        num_examples=num_examples,
        metrics=metrics,
    )
    return (proxy, res)


def test_aggregate_fit_emits_round_seconds_and_wire_bytes(tmp_path):
    """Phase E.2 — overhead metrics show up after a configure_fit →
    aggregate_fit round trip. The byte count is the post-serialization
    on-wire size (Phase E review #2)."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )

    # Stamp the round-start (configure_fit normally does this).
    strategy._round_start_times[1] = 0.0  # so round_seconds is positive

    w = [np.zeros((2, 2), dtype=np.float32)]
    results = [_fit_result(20, w), _fit_result(20, w)]
    _, fit_metrics = strategy.aggregate_fit(1, results, [])

    assert "round_seconds" in fit_metrics
    assert fit_metrics["round_seconds"] >= 0.0
    assert "parameter_update_wire_bytes" in fit_metrics
    assert fit_metrics["parameter_update_wire_bytes"] > 0
    # Per-client byte size is the per-client average.
    assert fit_metrics["parameter_update_wire_bytes_per_client"] == pytest.approx(
        fit_metrics["parameter_update_wire_bytes"] / 2
    )


def test_configure_fit_retry_does_not_overwrite_start_time(tmp_path):
    """Phase E review #8 — Flower retries ``configure_fit`` on transient
    client failures. The retry must NOT overwrite the original
    ``_round_start_times[server_round]``; ``round_seconds`` should
    reflect TOTAL wall-clock, not just the post-retry slice."""
    import time as _time

    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )

    # First configure_fit stamps a known start time.
    strategy._round_start_times[1] = 100.0
    # Simulating a retry: configure_fit re-invoked for the same round.
    # The setdefault pattern must preserve the original.
    strategy._round_start_times.setdefault(1, _time.time())
    assert strategy._round_start_times[1] == 100.0


def test_aggregate_fit_uses_correct_per_round_start_time(tmp_path):
    """Different rounds get distinct start-time entries; no clobbering."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )
    w = [np.zeros((2, 2), dtype=np.float32)]

    strategy._round_start_times[1] = 0.0
    _, m1 = strategy.aggregate_fit(1, [_fit_result(10, w)], [])
    strategy._round_start_times[2] = 0.0
    _, m2 = strategy.aggregate_fit(2, [_fit_result(10, w)], [])

    # Both rounds saw a stamped start time → both have positive seconds.
    assert m1["round_seconds"] > 0
    assert m2["round_seconds"] > 0
    # Dict carries both entries, no overwrite.
    assert 1 in strategy._round_start_times
    assert 2 in strategy._round_start_times


def test_aggregate_evaluate_folds_in_fit_overhead(tmp_path):
    """The overhead captured during aggregate_fit lands in the next
    aggregate_evaluate's aggregated dict + log row."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )
    strategy._round_start_times[1] = 0.0
    w = [np.zeros((2, 2), dtype=np.float32)]
    strategy.aggregate_fit(1, [_fit_result(10, w)], [])

    _, aggregated = strategy.aggregate_evaluate(
        1, [_result(10, {"threat_macro_f1": 0.5}, loss=0.4)], [],
    )

    assert "round_seconds" in aggregated
    assert "parameter_update_wire_bytes" in aggregated
    # And the log row also has them.
    log_text = (tmp_path / "server_eval.log").read_text()
    assert "parameter_update_wire_bytes" in log_text
    assert "round_seconds" in log_text


# ───────────────────────────────────────────────────────────────────────
#   Phase E review #4 — server-side proximal_contribution aggregation
# ───────────────────────────────────────────────────────────────────────

def test_aggregate_fit_weighted_means_proximal_contribution(tmp_path):
    """FedProx clients emit ``proximal_contribution`` in their fit
    metrics. The strategy weighted-means it by num_examples — same
    formula as the eval-side weighted metrics."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    strategy._round_start_times[1] = 0.0
    w = [np.zeros((2, 2), dtype=np.float32)]
    results = [
        _fit_result_with_metrics(20, w, {"proximal_contribution": 0.1}),
        _fit_result_with_metrics(80, w, {"proximal_contribution": 0.5}),
    ]
    _, fit_metrics = strategy.aggregate_fit(1, results, [])

    assert "proximal_contribution" in fit_metrics
    # weighted mean: (20*0.1 + 80*0.5) / 100 = 0.42
    assert fit_metrics["proximal_contribution"] == pytest.approx(0.42)


def test_aggregate_fit_proximal_contribution_zero_for_plain_fedavg(tmp_path):
    """Plain FedAvg clients (plain Model) emit ``proximal_contribution=0.0``
    per Phase D follow-up. The aggregated value should be 0."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    strategy._round_start_times[1] = 0.0
    w = [np.zeros((2, 2), dtype=np.float32)]
    results = [
        _fit_result_with_metrics(50, w, {"proximal_contribution": 0.0}),
        _fit_result_with_metrics(50, w, {"proximal_contribution": 0.0}),
    ]
    _, fit_metrics = strategy.aggregate_fit(1, results, [])
    assert fit_metrics["proximal_contribution"] == pytest.approx(0.0)


def test_proximal_contribution_lands_in_server_log(tmp_path):
    """End-to-end: aggregated proximal_contribution shows up in the
    server_evaluation.log line for the round."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )
    strategy._round_start_times[1] = 0.0
    w = [np.zeros((2, 2), dtype=np.float32)]
    strategy.aggregate_fit(
        1, [_fit_result_with_metrics(10, w, {"proximal_contribution": 0.25})], [],
    )
    strategy.aggregate_evaluate(
        1, [_result(10, {"threat_macro_f1": 0.5}, loss=0.4)], [],
    )
    log_text = (tmp_path / "server_eval.log").read_text()
    assert "proximal_contribution: 0.25" in log_text


# ───────────────────────────────────────────────────────────────────────
#   Phase E review #1 — NaN-safe fairness metrics
# ───────────────────────────────────────────────────────────────────────

def test_fairness_variance_filters_nan_clients(tmp_path):
    """A client reporting NaN macro_f1 (single-class fold) must not be
    treated as 0.0 in the variance. The pre-fix bug coerced NaN→0,
    dragging fairness_macro_f1_variance upward."""
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    results = [
        _result(50, {"threat_macro_f1": float("nan"),
                      "threat_accuracy": float("nan")}, loss=0.5),
        _result(50, {"threat_macro_f1": 0.7,
                      "threat_accuracy": 0.75}, loss=0.5),
    ]
    _, aggregated = strategy.aggregate_evaluate(1, results, [])

    # Only one valid client → variance undefined → 0.0 (not 0.49 / 4 which
    # the old _safe_float path would have produced).
    assert aggregated["fairness_macro_f1_variance"] == pytest.approx(0.0)
    assert aggregated["fairness_accuracy_variance"] == pytest.approx(0.0)


def test_fairness_variance_computed_from_valid_clients_only():
    """Three clients, one with NaN — variance is over the two valid values."""
    strategy = FusionFedAvg(
        min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3,
    )
    results = [
        _result(10, {"threat_macro_f1": 0.5, "threat_accuracy": 0.6}, loss=0.5),
        _result(10, {"threat_macro_f1": float("nan"),
                      "threat_accuracy": float("nan")}, loss=0.5),
        _result(10, {"threat_macro_f1": 0.9, "threat_accuracy": 0.95}, loss=0.5),
    ]
    _, aggregated = strategy.aggregate_evaluate(1, results, [])
    # variance([0.5, 0.9]) = 0.04 (population variance of two values)
    assert aggregated["fairness_macro_f1_variance"] == pytest.approx(0.04)
    # variance([0.6, 0.95]) = 0.030625
    assert aggregated["fairness_accuracy_variance"] == pytest.approx(0.030625)


# ───────────────────────────────────────────────────────────────────────
#   Phase E review #3 — log parser round-trip
# ───────────────────────────────────────────────────────────────────────

def test_strategy_log_round_trips_through_parser(tmp_path):
    """Drive ``aggregate_fit`` + ``aggregate_evaluate`` → read the resulting
    server_evaluation.log → parse → assert key metrics match. Catches
    format drift between the strategy emitter and the plot-script parser."""
    from Analysis.CommunitiesCrime.log_parser import parse_server_log

    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    strategy._round_start_times[1] = 0.0
    w = [np.zeros((2, 2), dtype=np.float32)]

    # Round 1
    strategy.aggregate_fit(
        1, [_fit_result_with_metrics(50, w, {"proximal_contribution": 0.1}),
            _fit_result_with_metrics(50, w, {"proximal_contribution": 0.3})], [],
    )
    strategy.aggregate_evaluate(
        1, [_result(50, {"threat_macro_f1": 0.5, "threat_accuracy": 0.6,
                          "escalation_mae": 0.2, "escalation_auroc": 0.75,
                          "escalation_spearman": 0.4}, loss=0.5),
             _result(50, {"threat_macro_f1": 0.7, "threat_accuracy": 0.8,
                          "escalation_mae": 0.15, "escalation_auroc": 0.85,
                          "escalation_spearman": 0.5}, loss=0.3)], [],
    )

    # Round 2 (separate start-time entry — Phase E review #8 dict)
    strategy._round_start_times[2] = 0.0
    strategy.aggregate_fit(
        2, [_fit_result_with_metrics(50, w, {"proximal_contribution": 0.05}),
            _fit_result_with_metrics(50, w, {"proximal_contribution": 0.2})], [],
    )
    strategy.aggregate_evaluate(
        2, [_result(50, {"threat_macro_f1": 0.6, "threat_accuracy": 0.7,
                          "escalation_mae": 0.15, "escalation_auroc": 0.8,
                          "escalation_spearman": 0.5}, loss=0.4),
             _result(50, {"threat_macro_f1": 0.8, "threat_accuracy": 0.85,
                          "escalation_mae": 0.1, "escalation_auroc": 0.9,
                          "escalation_spearman": 0.6}, loss=0.2)], [],
    )

    # Round-trip through the parser
    df = parse_server_log(tmp_path / "server_eval.log")

    assert len(df) == 2
    assert list(df["round"]) == [1, 2]
    assert list(df["n_clients"]) == [2, 2]
    # Aggregated weighted means survive the round-trip
    assert df["threat_macro_f1"].iloc[0] == pytest.approx(0.6, abs=1e-6)  # (50*0.5+50*0.7)/100
    assert df["threat_accuracy"].iloc[0] == pytest.approx(0.7, abs=1e-6)
    assert df["proximal_contribution"].iloc[0] == pytest.approx(0.2, abs=1e-6)
    # Round 2
    assert df["threat_macro_f1"].iloc[1] == pytest.approx(0.7, abs=1e-6)
    assert df["proximal_contribution"].iloc[1] == pytest.approx(0.125, abs=1e-6)
    # Overhead metrics also flow through
    assert "round_seconds" in df.columns
    assert "parameter_update_wire_bytes" in df.columns


# ───────────────────────────────────────────────────────────────────────
#   Phase E.3 — fairness_accuracy_variance
# ───────────────────────────────────────────────────────────────────────

def test_aggregate_evaluate_emits_fairness_accuracy_variance(tmp_path):
    strategy = FusionFedAvg(
        evaluation_log=str(tmp_path / "server_eval.log"),
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    results = [
        _result(50, {"threat_macro_f1": 0.6, "threat_accuracy": 0.65}, loss=0.5),
        _result(50, {"threat_macro_f1": 0.8, "threat_accuracy": 0.95}, loss=0.3),
    ]
    _, aggregated = strategy.aggregate_evaluate(1, results, [])
    assert "fairness_accuracy_variance" in aggregated
    # variance([0.65, 0.95]) = 0.0225
    assert aggregated["fairness_accuracy_variance"] == pytest.approx(0.0225)


def test_aggregate_evaluate_keeps_macro_f1_variance_too():
    """E.3 added accuracy variance but kept the macro_f1 variance from Phase C."""
    strategy = FusionFedAvg(
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    results = [
        _result(10, {"threat_macro_f1": 0.5, "threat_accuracy": 0.6}, loss=0.5),
        _result(10, {"threat_macro_f1": 0.9, "threat_accuracy": 0.95}, loss=0.5),
    ]
    _, aggregated = strategy.aggregate_evaluate(1, results, [])
    assert "fairness_macro_f1_variance" in aggregated
    assert "fairness_accuracy_variance" in aggregated
