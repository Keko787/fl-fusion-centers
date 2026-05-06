"""Build a publication-clean R* regression figure + supporting table.

The default ``exp1_Rstar_regression.png`` plots one line per (Dpd, alpha)
cell, but FL processing time depends only on R (and constant
theta_bytes / B_actual), so all six lines collapse onto one curve. That
is the *substantive finding* of the experiment, but the figure ends up
visually confusing.

This module replaces it with three complementary artifacts:

* ``exp1_Rstar_regression_clean.png`` --- two lines (one per Dpd,
  averaged over alpha) plus the centralized baseline. The two Dpd lines
  are within 0.04 s of each other but plotted on a zoomed inset so the
  small Dpd offset is visible.
* ``exp1_Rstar_residuals.png`` --- residual plot of Tproc(R) minus the
  per-cell linear fit, showing measurement variance is < 0.05 s.
* ``exp1_Rstar_per_cell_means.csv`` --- the 18-row table (6 cells x 3 R
  values) suitable for direct inclusion in the paper.

Runnable directly::

    python -m experiments.analysis.exp1_rstar_clean \\
        --csv results/exp1_chameleon_no1gb.csv \\
        --out-dir DeveloperDocs/figures/exp1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis.exp1 import _parse_size, load_trials


def _per_cell_means(rows):
    """Return {(Dpd, alpha, R): mean Tproc} for FL ok rows only."""
    fl = [r for r in rows if r.is_ok and r.arm == "FL"]
    keys = sorted({(r.Dpd, r.alpha, r.R) for r in fl},
                  key=lambda k: (_parse_size(k[0]), k[1], k[2]))
    return {
        (Dpd, alpha, R): float(np.mean([
            r.Tproc_s for r in fl
            if r.Dpd == Dpd and r.alpha == alpha and r.R == R
        ]))
        for (Dpd, alpha, R) in keys
    }


def _per_dpd_means(per_cell):
    """Average across alpha to give {(Dpd, R): mean Tproc}."""
    out: dict[tuple[str, int], float] = {}
    Dpds = sorted({k[0] for k in per_cell}, key=_parse_size)
    Rs = sorted({k[2] for k in per_cell})
    for Dpd in Dpds:
        for R in Rs:
            vals = [v for (d, a, r), v in per_cell.items()
                    if d == Dpd and r == R]
            out[(Dpd, R)] = float(np.mean(vals))
    return out, Dpds, Rs


def _centralized_mean(rows):
    cent = [r.Tproc_s for r in rows if r.is_ok and r.arm == "Centralized"]
    return float(np.mean(cent)) if cent else float("nan")


def _write_table(per_cell, out_path: Path) -> None:
    rows_out = []
    for (Dpd, alpha, R), v in sorted(
        per_cell.items(),
        key=lambda kv: (_parse_size(kv[0][0]), kv[0][1], kv[0][2]),
    ):
        rows_out.append({"Dpd": Dpd, "alpha": alpha, "R": R,
                          "Tproc_s_mean": f"{v:.4f}"})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Dpd", "alpha", "R", "Tproc_s_mean"])
        w.writeheader()
        for row in rows_out:
            w.writerow(row)
    print(f"wrote {out_path}")


def _plot_clean(per_dpd, Dpds, Rs, cent_mean, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"10MB": "tab:blue", "100MB": "tab:orange"}
    for Dpd in Dpds:
        ys = [per_dpd[(Dpd, R)] for R in Rs]
        ax.plot(Rs, ys, marker="o", linewidth=2,
                label=f"FL, Dpd={Dpd} (mean over alpha)",
                color=colors.get(Dpd, None))
    ax.axhline(cent_mean, linestyle="--", color="black",
               label=f"Centralized mean baseline ({cent_mean:.2f} s)")

    slope = (per_dpd[(Dpds[0], Rs[-1])] - per_dpd[(Dpds[0], Rs[0])]) / (Rs[-1] - Rs[0])
    R_star_proj = cent_mean / slope if slope > 0 else float("nan")
    ax.axvline(R_star_proj, linestyle=":", color="grey",
               label=f"projected R* = {R_star_proj:.1f}")

    ax.set_xlabel("R (FL rounds)")
    ax.set_ylabel("T_proc (s)")
    ax.set_title("FL T_proc vs R --- collapsed across alpha, separated by Dpd")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.85),
              fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.3)

    # Inset zoom on R=5 to make the Dpd offset visible.
    axins = ax.inset_axes((0.55, 0.18, 0.4, 0.32))
    for Dpd in Dpds:
        axins.plot([Rs[0]], [per_dpd[(Dpd, Rs[0])]], marker="o",
                    markersize=10, color=colors.get(Dpd, None))
        axins.annotate(
            f"{Dpd}: {per_dpd[(Dpd, Rs[0])]:.4f}s",
            xy=(Rs[0], per_dpd[(Dpd, Rs[0])]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, va="center",
        )
    axins.set_xlim(Rs[0] - 0.5, Rs[0] + 1.5)
    ymin = min(per_dpd[(d, Rs[0])] for d in Dpds) - 0.02
    ymax = max(per_dpd[(d, Rs[0])] for d in Dpds) + 0.02
    axins.set_ylim(ymin, ymax)
    axins.set_title("zoom: R=5", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_residuals(per_cell, out_path: Path) -> None:
    """Plot Tproc(R) - linear_fit(R) per (Dpd, alpha) cell."""
    cells = sorted({(d, a) for (d, a, _) in per_cell},
                    key=lambda k: (_parse_size(k[0]), k[1]))
    Rs = sorted({r for (_, _, r) in per_cell})

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for Dpd, alpha in cells:
        ys = np.array([per_cell[(Dpd, alpha, R)] for R in Rs])
        # Linear fit through origin (per-round wire time).
        slope = np.sum(np.array(Rs) * ys) / np.sum(np.array(Rs) ** 2)
        residuals = ys - slope * np.array(Rs)
        ax.plot(Rs, residuals * 1000.0, marker="o",
                label=f"Dpd={Dpd}, alpha={alpha}")

    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("R (FL rounds)")
    ax.set_ylabel("residual T_proc - slope*R  (ms)")
    ax.set_title("Per-cell residuals from linear fit (slope * R)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.analysis.exp1_rstar_clean")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    rows = load_trials(args.csv)
    per_cell = _per_cell_means(rows)
    per_dpd, Dpds, Rs = _per_dpd_means(per_cell)
    cent_mean = _centralized_mean(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_table(per_cell, args.out_dir / "exp1_Rstar_per_cell_means.csv")
    _plot_clean(per_dpd, Dpds, Rs, cent_mean,
                args.out_dir / "exp1_Rstar_regression_clean.png")
    _plot_residuals(per_cell,
                     args.out_dir / "exp1_Rstar_residuals.png")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
