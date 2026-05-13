"""Generate the three result figures referenced by paper/main.tex.

Figures produced (under ``paper/Figures/`` by default):

  * ``per_client_distribution_geo_n5.png``  — bar chart of the geographic
    partition (Group A motivation for the FedAvg-vs-FedProx comparison).
  * ``headline_convergence.png``             — overlaid macro-F1 vs. rounds
    for FedAvg-IID, FedAvg-Geographic, FedProx-Geographic with the
    centralized baseline as a horizontal reference line.
  * ``proximal_evolution.png``               — FedProx proximal-term
    contribution decay across rounds.

The script auto-detects which run directories are available and plots
real data where it can. For any figure whose source run is missing, it
generates a clearly-watermarked synthetic placeholder so the paper
compiles before the experiments finish.

Search order for each run (first hit wins):

  per-client distribution — geographic partition
    1. results/exp4_fedavg_geo_n3      (N=3 deployment)
    2. results/exp4_fedavg_geo         (N=5 default)
    3. results/results_1/exp4_fedavg_geo (legacy N=5 run)

  headline convergence — three FL runs
    FedAvg-IID  : results/exp3_fedavg_iid_n3,  results/exp3_fedavg_iid,  results/results_1/exp3_fedavg_iid
    FedAvg-geo  : results/exp4_fedavg_geo_n3,  results/exp4_fedavg_geo,  results/results_1/exp4_fedavg_geo
    FedProx-geo : results/exp5_fedprox_geo_n3, results/exp5_fedprox_geo, results/results_1/exp5_fedprox_geo

  proximal evolution — FedProx run
    results/exp5_fedprox_geo_n3, results/exp5_fedprox_geo, results/results_1/exp5_fedprox_geo

The centralized baseline value for the headline reference line is read
from the first available ``results/{exp1_centralized,results_1/exp1_centralized}/<ts>_evaluation.log``.
If no centralized log is found, the value falls back to 0.62.

Usage::

    # From project root (with venv active)
    python paper/generate_figures.py                 # auto: real where available, synthetic otherwise
    python paper/generate_figures.py --mode synthetic # force all synthetic (paper-compile test)
    python paper/generate_figures.py --mode real     # error if any source is missing

Re-run after every experiment finishes — figures backed by real data
will silently switch from the synthetic placeholder to the real plot.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Analysis.CommunitiesCrime import (
    plot_centralized_vs_federated,
    plot_per_client_distribution,
)

# ---------------------------------------------------------------------------
# Run-directory search tables
# ---------------------------------------------------------------------------
RESULTS = REPO_ROOT / "results"

GEO_RUN_CANDIDATES = [
    RESULTS / "exp4_fedavg_geo_n5",
    RESULTS / "exp4_fedavg_geo_n3",
    RESULTS / "exp4_fedavg_geo",
    RESULTS / "results_1" / "exp4_fedavg_geo",
]

IID_RUN_CANDIDATES = [
    RESULTS / "exp3_fedavg_iid_n5",
    RESULTS / "exp3_fedavg_iid_n3",
    RESULTS / "exp3_fedavg_iid",
    RESULTS / "results_1" / "exp3_fedavg_iid",
]

PROX_RUN_CANDIDATES = [
    RESULTS / "exp5_fedprox_geo_n5",
    RESULTS / "exp5_fedprox_geo_n3",
    RESULTS / "exp5_fedprox_geo",
    RESULTS / "results_1" / "exp5_fedprox_geo",
]

CENTRAL_RUN_CANDIDATES = [
    RESULTS / "exp1_centralized",
    RESULTS / "results_1" / "exp1_centralized",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def first_valid(candidates, marker: str) -> Optional[Path]:
    """Return the first candidate path that contains *marker*, else None."""
    for c in candidates:
        if (c / marker).is_file():
            return c
    return None


def read_centralized_baseline(default: float = 0.62) -> float:
    """Pull the final threat_macro_f1 from the most recent centralized run."""
    for run_dir in CENTRAL_RUN_CANDIDATES:
        for log in sorted(run_dir.glob("*_evaluation.log"), reverse=True):
            m = re.search(r"threat_macro_f1:\s*([\d.]+)", log.read_text())
            if m:
                value = float(m.group(1))
                print(f"  centralized baseline = {value:.4f}  (from {log.relative_to(REPO_ROOT)})")
                return value
    print(f"  centralized baseline = {default} (default — no exp1 log found)")
    return default


def stamp_synthetic(ax, note: str = "SYNTHETIC PLACEHOLDER") -> None:
    """Drop a clear diagonal watermark on synthetic figures."""
    ax.text(
        0.5, 0.5, note,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=22, color="red", alpha=0.18,
        rotation=30, weight="bold",
    )


# ---------------------------------------------------------------------------
# Per-client distribution
# ---------------------------------------------------------------------------
def make_per_client_distribution(out: Path, mode: str) -> str:
    """Generate per_client_distribution_geo_n5.png. Returns source description."""
    geo = None if mode == "synthetic" else first_valid(GEO_RUN_CANDIDATES, "partition_stats.json")

    if geo is not None:
        plot_per_client_distribution.plot(geo, output=out, style="paper")
        return f"real from {geo.relative_to(REPO_ROOT)}"

    if mode == "real":
        raise FileNotFoundError("No geographic run with partition_stats.json found.")

    # Synthetic: hand-rolled non-IID bar chart at N=5
    rng = np.random.default_rng(42)
    n_clients = 5
    base = np.array([[120, 80, 30], [100, 90, 40], [60, 100, 70],
                     [40, 110, 90], [30, 90, 130]])
    base = base + rng.integers(-10, 11, base.shape)
    class_labels = ["violent", "property", "other"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(n_clients)
    bottom = np.zeros(n_clients)
    for i, lbl in enumerate(class_labels):
        ax.bar(x, base[:, i], bottom=bottom, label=lbl)
        bottom += base[:, i]
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Train samples")
    ax.set_title("Per-client class distribution (strategy=geographic, N=5)")
    ax.set_xticks(x, [str(i) for i in range(n_clients)])
    ax.legend(title="threat_class")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stamp_synthetic(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "synthetic placeholder"


# ---------------------------------------------------------------------------
# Headline convergence
# ---------------------------------------------------------------------------
def synth_curve(start: float, end: float, rounds: int, noise: float,
                seed: int) -> np.ndarray:
    """Smooth saturating curve from *start* to *end* with seeded noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, rounds)
    base = start + (end - start) * (1 - np.exp(-3 * t))
    return base + rng.normal(0, noise, rounds)


def make_headline_convergence(out: Path, mode: str, baseline: float) -> str:
    """Generate headline_convergence.png. Returns source description."""
    available = {}
    sources = []
    if mode != "synthetic":
        for label, candidates in [
            ("FedAvg-IID", IID_RUN_CANDIDATES),
            ("FedAvg-Geographic", GEO_RUN_CANDIDATES),
            ("FedProx-Geographic", PROX_RUN_CANDIDATES),
        ]:
            run = first_valid(candidates, "server_evaluation.log")
            if run is not None:
                available[label] = run
                sources.append(f"{label}={run.relative_to(REPO_ROOT)}")

    if mode == "real" and len(available) < 3:
        raise FileNotFoundError(
            f"--mode real requires all 3 runs; missing: "
            f"{set(['FedAvg-IID', 'FedAvg-Geographic', 'FedProx-Geographic']) - set(available)}"
        )

    # If we have all three runs, use the official plot module
    if len(available) == 3:
        plot_centralized_vs_federated.plot(
            available, output=out,
            centralized_baseline=baseline, style="paper",
        )
        return "real from " + "; ".join(sources)

    # Otherwise build a custom plot: real curves where we have them,
    # synthetic curves for the rest.
    from Analysis.CommunitiesCrime.log_parser import collect_server_logs

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rounds = 50

    expected = {
        "FedAvg-IID":         dict(start=0.32, end=0.60, noise=0.012, seed=11),
        "FedAvg-Geographic":  dict(start=0.30, end=0.55, noise=0.020, seed=22),
        "FedProx-Geographic": dict(start=0.30, end=0.58, noise=0.012, seed=33),
    }

    note_parts = []
    if available:
        df = collect_server_logs(list(available.values()), list(available.keys()))
        for label, sub in df.groupby("label"):
            if "threat_macro_f1" not in sub.columns:
                continue
            ax.plot(sub["round"], sub["threat_macro_f1"],
                    label=f"{label} (real)", marker="o", markersize=3)
            note_parts.append(f"{label}=real")

    for label, params in expected.items():
        if label in available:
            continue
        y = synth_curve(rounds=rounds, **params)
        ax.plot(np.arange(1, rounds + 1), y,
                label=f"{label} (synthetic)", linestyle="--", alpha=0.7)
        note_parts.append(f"{label}=synth")

    ax.axhline(baseline, linestyle=":", color="black", alpha=0.7,
               label=f"Centralized ({baseline:.3f})")
    ax.set_xlabel("Round")
    ax.set_ylabel("threat_macro_f1")
    ax.set_title("Headline convergence — centralized vs federated")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if any("synth" in n for n in note_parts):
        stamp_synthetic(ax, "PARTIAL SYNTHETIC")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "mixed: " + ", ".join(note_parts)


# ---------------------------------------------------------------------------
# Proximal evolution
# ---------------------------------------------------------------------------
def make_proximal_evolution(out: Path, mode: str) -> str:
    """Generate proximal_evolution.png. Returns source description."""
    prox = None if mode == "synthetic" else first_valid(PROX_RUN_CANDIDATES, "server_evaluation.log")

    if prox is not None:
        plot_centralized_vs_federated.plot(
            {"FedProx-Geographic": prox},
            output=out, metric="proximal_contribution", style="paper",
        )
        return f"real from {prox.relative_to(REPO_ROOT)}"

    if mode == "real":
        raise FileNotFoundError("No FedProx run with server_evaluation.log found.")

    # Synthetic: exponential decay characteristic of FedProx proximal term
    # as the global model converges.
    rng = np.random.default_rng(7)
    rounds = 50
    t = np.arange(1, rounds + 1)
    # Decay from ~0.13 at round 1 toward ~0.04 by round 50
    pi = 0.04 + 0.09 * np.exp(-t / 12.0) + rng.normal(0, 0.004, rounds)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(t, pi, marker="o", markersize=3, label="FedProx-Geographic")
    ax.set_xlabel("Round")
    ax.set_ylabel(r"proximal_contribution  $\Pi = (\mu/2)\sum_j \|\theta_j - \theta^t\|^2$")
    ax.set_title(r"Proximal-term decay under FedProx ($\mu = 0.01$)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stamp_synthetic(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "synthetic placeholder"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the three result figures referenced by paper/main.tex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent / "Figures"),
        help="Output directory (default: paper/Figures).",
    )
    parser.add_argument(
        "--mode", choices=["auto", "real", "synthetic"], default="auto",
        help="auto = real where available (default); real = error on missing source; "
             "synthetic = force placeholders regardless of run availability.",
    )
    parser.add_argument(
        "--centralized-baseline", type=float, default=None,
        help="Override centralized macro-F1 reference line (default: read from "
             "exp1's evaluation log, falling back to 0.62).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures into {out_dir.relative_to(REPO_ROOT)} (mode={args.mode})")
    baseline = args.centralized_baseline if args.centralized_baseline is not None \
        else read_centralized_baseline()

    manifest = {}
    tasks = [
        ("per_client_distribution_geo_n5.png",
         lambda p: make_per_client_distribution(p, args.mode)),
        ("headline_convergence.png",
         lambda p: make_headline_convergence(p, args.mode, baseline)),
        ("proximal_evolution.png",
         lambda p: make_proximal_evolution(p, args.mode)),
    ]
    for name, fn in tasks:
        out = out_dir / name
        source = fn(out)
        manifest[name] = source
        print(f"  [OK] {name:<42}  {source}")

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {(out_dir / 'MANIFEST.json').relative_to(REPO_ROOT)}.")
    print("Re-run after each experiment completes; figures backed by real data will switch automatically.")


if __name__ == "__main__":
    main()
