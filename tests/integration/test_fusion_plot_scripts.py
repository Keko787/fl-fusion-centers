"""Phase E.4 — plot-script smoke tests.

Each test fabricates synthetic logs/artifacts under a tmp dir, runs
one of the plot scripts, and asserts a non-empty PNG landed at the
expected path. These are smoke tests — they verify the plot pipeline
doesn't crash on plausible inputs, not that the images look right.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import pytest

from Analysis.CommunitiesCrime.plot_centralized_vs_federated import (
    plot as plot_centralized_vs_federated,
)
from Analysis.CommunitiesCrime.plot_per_client_distribution import (
    plot as plot_per_client_distribution,
)
from Analysis.CommunitiesCrime.plot_scaling_n_clients import (
    plot as plot_scaling_n_clients,
)


_SERVER_LOG_TEMPLATE = """=== Round {r} (5 clients) ===
aggregated_loss: {loss}
threat_macro_f1: {f1}
escalation_mae: 0.15
escalation_auroc: 0.75
escalation_spearman: 0.45
fairness_macro_f1_variance: 0.01
fairness_accuracy_variance: 0.008
round_seconds: 2.0
parameter_update_wire_bytes: 50000
"""


def _make_server_log(run_dir: Path, n_rounds: int = 10,
                      start_f1: float = 0.4, end_f1: float = 0.7) -> Path:
    """Write a fake server_evaluation.log with smoothly improving macro-F1."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "server_evaluation.log"
    blocks = []
    for r in range(1, n_rounds + 1):
        progress = (r - 1) / max(1, n_rounds - 1)
        f1 = start_f1 + (end_f1 - start_f1) * progress
        loss = 0.6 - 0.3 * progress
        blocks.append(_SERVER_LOG_TEMPLATE.format(r=r, loss=loss, f1=f1))
    log.write_text("\n".join(blocks))
    return log


def _make_partition_stats(run_dir: Path, n_clients: int = 5) -> Path:
    """Write a fake partition_stats.json with mock class distributions."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "strategy": "geographic",
        "num_clients": n_clients,
        "seed": 42,
        "clients": {
            str(cid): {
                "train": {
                    "n_rows": 100 + 10 * cid,
                    "class_distribution": {
                        "0": 60 - 5 * cid,
                        "1": 30,
                        "2": 10 + 5 * cid,
                    },
                },
                "val": {"n_rows": 20, "class_distribution": {}},
            }
            for cid in range(n_clients)
        },
    }
    f = run_dir / "partition_stats.json"
    f.write_text(json.dumps(stats))
    return f


# ───────────────────────────────────────────────────────────────────────
#   plot_per_client_distribution
# ───────────────────────────────────────────────────────────────────────

def test_plot_per_client_distribution_smoke(tmp_path):
    run_dir = tmp_path / "run"
    _make_partition_stats(run_dir, n_clients=5)
    path, fig = plot_per_client_distribution(run_dir)
    assert path.exists()
    assert path.stat().st_size > 0  # PNG has content
    assert fig is not None


def test_plot_per_client_distribution_custom_output(tmp_path):
    run_dir = tmp_path / "run"
    _make_partition_stats(run_dir, n_clients=3)
    out = tmp_path / "custom_name.png"
    path, _ = plot_per_client_distribution(run_dir, output=out)
    assert path == out
    assert out.exists()


def test_plot_per_client_distribution_empty_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "partition_stats.json").write_text(
        json.dumps({"strategy": "iid", "num_clients": 0, "clients": {}})
    )
    with pytest.raises(ValueError, match="No client entries"):
        plot_per_client_distribution(run_dir)


def test_plot_per_client_distribution_paper_style(tmp_path):
    run_dir = tmp_path / "run"
    _make_partition_stats(run_dir, n_clients=3)
    path, _ = plot_per_client_distribution(run_dir, style="paper")
    assert path.exists()


def test_plot_per_client_distribution_pdf_output(tmp_path):
    """Vector output works via the file extension (Phase E review #11)."""
    run_dir = tmp_path / "run"
    _make_partition_stats(run_dir, n_clients=3)
    out = tmp_path / "vector.pdf"
    path, _ = plot_per_client_distribution(run_dir, output=out)
    assert path == out
    assert out.exists()
    # PDF magic bytes start with %PDF
    assert out.read_bytes().startswith(b"%PDF")


def test_plot_returns_reusable_figure(tmp_path):
    """Phase E review #10 — plot returns the Figure when ``close=False``,
    so notebooks can re-render it without reading the file back."""
    run_dir = tmp_path / "run"
    _make_partition_stats(run_dir, n_clients=3)
    path, fig = plot_per_client_distribution(run_dir, close=False)
    # Figure is still alive (axes still present, not yet closed).
    assert len(fig.axes) > 0


# ───────────────────────────────────────────────────────────────────────
#   plot_centralized_vs_federated
# ───────────────────────────────────────────────────────────────────────

def test_plot_centralized_vs_federated_smoke(tmp_path):
    fedavg_iid = tmp_path / "fedavg_iid"
    fedavg_geo = tmp_path / "fedavg_geo"
    fedprox_geo = tmp_path / "fedprox_geo"
    _make_server_log(fedavg_iid, end_f1=0.70)
    _make_server_log(fedavg_geo, end_f1=0.62)
    _make_server_log(fedprox_geo, end_f1=0.66)

    out = tmp_path / "headline.png"
    path, fig = plot_centralized_vs_federated(
        runs={"FedAvg-IID": fedavg_iid,
              "FedAvg-geo": fedavg_geo,
              "FedProx-geo": fedprox_geo},
        output=out,
        centralized_baseline=0.72,
    )
    assert path == out
    assert out.exists()
    assert out.stat().st_size > 0
    assert fig is not None


def test_plot_centralized_vs_federated_default_output(tmp_path, monkeypatch):
    """No --output → defaults to ./centralized_vs_federated.png in CWD."""
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    _make_server_log(run_dir)
    path, _ = plot_centralized_vs_federated(runs={"R": run_dir})
    assert path.name == "centralized_vs_federated.png"
    assert path.exists()


def test_plot_centralized_vs_federated_no_runs_raises(tmp_path):
    with pytest.raises(ValueError, match="No runs"):
        plot_centralized_vs_federated(runs={}, output=tmp_path / "x.png")


def test_plot_centralized_vs_federated_skips_missing_metric(tmp_path):
    """If a run's log lacks the requested metric, the plot still works
    (just no line for that run)."""
    run_dir = tmp_path / "minimal"
    run_dir.mkdir()
    (run_dir / "server_evaluation.log").write_text(
        "=== Round 1 (1 clients) ===\naggregated_loss: 0.5\n"
    )
    out = tmp_path / "out.png"
    plot_centralized_vs_federated(
        runs={"minimal": run_dir}, output=out, metric="threat_macro_f1",
    )
    assert out.exists()


# ───────────────────────────────────────────────────────────────────────
#   plot_scaling_n_clients
# ───────────────────────────────────────────────────────────────────────

def test_plot_scaling_n_clients_smoke(tmp_path):
    for n in (3, 5, 10):
        _make_server_log(tmp_path / f"n{n}", end_f1=0.6 + 0.02 * (10 - n))

    out = tmp_path / "scaling.png"
    path, fig = plot_scaling_n_clients(
        runs={3: tmp_path / "n3", 5: tmp_path / "n5", 10: tmp_path / "n10"},
        output=out,
    )
    assert path == out
    assert out.exists()
    assert fig is not None


def test_plot_scaling_n_clients_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No runs"):
        plot_scaling_n_clients(runs={}, output=tmp_path / "x.png")


# ───────────────────────────────────────────────────────────────────────
#   Phase E review #5 + #6 — smoothing + best-of-smoothed
# ───────────────────────────────────────────────────────────────────────

def test_smoothed_final_metric_ignores_last_round_overshoot(tmp_path):
    """A run that peaks mid-training then declines at the very last
    round should NOT have its bar height reflect the post-decline
    value. The best-smoothed selection picks the peak."""
    from Analysis.CommunitiesCrime.plot_scaling_n_clients import _best_smoothed
    from Analysis.CommunitiesCrime.log_parser import parse_server_log

    run_dir = tmp_path / "overshoot"
    run_dir.mkdir()
    # Hand-build a log where macro-F1 peaks at round 6 then declines.
    blocks = []
    f1_curve = [0.4, 0.5, 0.6, 0.7, 0.75, 0.78, 0.8, 0.78, 0.76, 0.5]  # last-round overshoot down
    for r, f1 in enumerate(f1_curve, start=1):
        blocks.append(
            f"=== Round {r} (5 clients) ===\n"
            f"aggregated_loss: 0.5\n"
            f"threat_macro_f1: {f1}\n"
        )
    (run_dir / "server_evaluation.log").write_text("\n".join(blocks))

    df = parse_server_log(run_dir / "server_evaluation.log")
    last = float(df["threat_macro_f1"].iloc[-1])
    best = _best_smoothed(df, "threat_macro_f1", smooth_window=3)

    # Last-round value is 0.5 (the dip); best smoothed should be near 0.78.
    assert last == pytest.approx(0.5)
    assert best > 0.7


def test_smoothed_convergence_robust_to_noise(tmp_path):
    """A single spike in an early round should NOT mark that round as
    'converged' — the rolling mean dampens it."""
    from Analysis.CommunitiesCrime.plot_scaling_n_clients import _rounds_to_convergence
    from Analysis.CommunitiesCrime.log_parser import parse_server_log

    run_dir = tmp_path / "noisy"
    run_dir.mkdir()
    # Round 5 is a lucky spike close to the eventual peak; real
    # convergence happens around round 25.
    f1_curve = (
        [0.40, 0.42, 0.44, 0.46, 0.78]  # round 5: lucky spike
        + [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66,
            0.68, 0.70, 0.72, 0.74, 0.76, 0.77, 0.78, 0.78, 0.79, 0.80]
    )
    blocks = [
        f"=== Round {r} (5 clients) ===\naggregated_loss: 0.5\nthreat_macro_f1: {f1}\n"
        for r, f1 in enumerate(f1_curve, start=1)
    ]
    (run_dir / "server_evaluation.log").write_text("\n".join(blocks))

    df = parse_server_log(run_dir / "server_evaluation.log")
    conv = _rounds_to_convergence(df, "threat_macro_f1", smooth_window=5)

    # Should NOT be round 5 (the noisy spike) — smoothing dampens it.
    assert conv > 5
