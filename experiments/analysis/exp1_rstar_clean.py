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


def _plot_with_cell_data(per_cell, per_dpd, Dpds, Rs, cent_mean, out_path):
    """Combined view: 18 raw cell means as markers + 2 trend lines.

    Each (Dpd, alpha) cell is drawn at a deterministic horizontal
    offset around its R value so all six co-located markers fan out
    visibly instead of stacking on top of each other (the cells differ
    by < 0.05 s at this y-scale, which is sub-pixel). The fan order
    reflects the schema:
      - colour = Dpd  (10MB blue, 100MB orange)
      - shape  = alpha  (circle 0.5, square 1.0, triangle 2.0)
      - x-offset = (Dpd group, alpha within group)

    A side-panel inset shows the un-offset numerical values so a reader
    who wants the actual T_proc per cell does not have to read offsets
    off the x-axis.
    """
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 5, wspace=0.05)
    ax = fig.add_subplot(gs[0, :4])
    ax_tab = fig.add_subplot(gs[0, 4])
    ax_tab.axis("off")

    Dpd_color = {"10MB": "#2b6cb0", "100MB": "#d97706"}
    alphas = sorted({a for (_, a, _) in per_cell})
    alpha_marker = {alphas[0]: "o", alphas[1]: "s", alphas[2]: "^"}

    # x-offset per (Dpd index, alpha index). Fan layout: 10MB cluster
    # left of the R tick, 100MB cluster right.
    cluster_half_width = 1.6
    intra_step = 0.5
    def _x_offset(Dpd: str, alpha: float) -> float:
        d_sign = -1.0 if Dpds.index(Dpd) == 0 else 1.0
        a_idx = alphas.index(alpha)
        a_off = (a_idx - (len(alphas) - 1) / 2.0) * intra_step
        return d_sign * cluster_half_width + a_off * d_sign * 0.0 + a_off

    for (Dpd, alpha, R), v in per_cell.items():
        ax.scatter(
            R + _x_offset(Dpd, alpha), v,
            color=Dpd_color.get(Dpd, "grey"),
            marker=alpha_marker.get(alpha, "x"),
            s=110, edgecolor="white", linewidth=0.8,
            zorder=4,
        )

    for Dpd in Dpds:
        ys = [per_dpd[(Dpd, R)] for R in Rs]
        ax.plot(Rs, ys, color=Dpd_color.get(Dpd, "grey"),
                 linewidth=2.2, alpha=0.85, zorder=2,
                 label=f"trend, Dpd = {Dpd}")

    ax.axhline(cent_mean, linestyle="--", color="black", linewidth=1.4,
                zorder=1, label=f"Centralized baseline = {cent_mean:.2f} s")

    slope = (per_dpd[(Dpds[0], Rs[-1])] - per_dpd[(Dpds[0], Rs[0])]) / (Rs[-1] - Rs[0])
    R_star_proj = cent_mean / slope if slope > 0 else float("nan")
    ax.axvline(R_star_proj, linestyle=":", color="#666666",
                linewidth=1.4, zorder=1)
    ax.annotate(
        f"projected R* = {R_star_proj:.1f}\nslope = {slope:.3f} s/round",
        xy=(R_star_proj, cent_mean * 0.45),
        xytext=(-12, 0), textcoords="offset points",
        ha="right", va="center", fontsize=10, color="#444444",
        bbox=dict(facecolor="white", edgecolor="#999999",
                   boxstyle="round,pad=0.4", linewidth=1.0),
    )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=Dpd_color[Dpds[0]], linewidth=2.2,
                label=f"trend, Dpd = {Dpds[0]} (mean over alpha)"),
        Line2D([0], [0], color=Dpd_color[Dpds[1]], linewidth=2.2,
                label=f"trend, Dpd = {Dpds[1]} (mean over alpha)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                label=f"Centralized baseline = {cent_mean:.2f} s"),
    ]
    handles += [
        Line2D([0], [0], marker=alpha_marker[a], color="grey",
                markerfacecolor="grey", markersize=10,
                markeredgecolor="white", linewidth=0,
                label=f"alpha = {a:g}")
        for a in alphas
    ]
    ax.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.01, 0.99), fontsize=9,
               framealpha=0.92, ncol=1)

    ax.set_xlabel("R (FL rounds)", fontsize=11)
    ax.set_ylabel(r"$T_{proc}$ (s)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(min(Rs) - 4, max(R_star_proj + 4, max(Rs) + 4))
    ax.set_ylim(0, max(cent_mean * 1.18, max(per_dpd.values()) * 1.10))

    # Side-panel CSV-as-table.
    ax_tab.set_title("per-cell means (s)", fontsize=10, loc="left")
    cell_rows = sorted(
        per_cell.items(),
        key=lambda kv: (_parse_size(kv[0][0]), kv[0][1], kv[0][2]),
    )
    table_data = [["Dpd", "alpha", "R", "Tproc"]]
    for (Dpd, alpha, R), v in cell_rows:
        table_data.append([Dpd, f"{alpha:g}", str(R), f"{v:.2f}"])
    tbl = ax_tab.table(
        cellText=table_data[1:], colLabels=table_data[0],
        loc="center", cellLoc="center",
        colWidths=[0.30, 0.20, 0.20, 0.30],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.15)
    # Colour-band the table rows by Dpd so the colour mapping carries
    # over from the scatter to the table.
    for i, (Dpd, _alpha, _R) in enumerate([row[0] for row in cell_rows]):
        for col in range(4):
            tbl[(i + 1, col)].set_facecolor(
                "#e6effa" if Dpd == Dpds[0] else "#fdf2e0"
            )

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
    _plot_with_cell_data(per_cell, per_dpd, Dpds, Rs, cent_mean,
                          args.out_dir / "exp1_Rstar_with_data.png")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
