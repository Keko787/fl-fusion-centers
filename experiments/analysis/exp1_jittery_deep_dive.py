"""Jittery vs primary deep-dive figures for Experiment 1.

The jittery ablation only ran one cell (Dpd=100MB, R=20, alpha=1.0),
so the standard heatmap / R*-regression / multi-Dpd energy-bar figures
are visually empty. But the same cell was also run on the primary link,
so we can do a direct primary-vs-jittery comparison --- the headline
finding for the jittery scenario:

    Centralized wall-time inflates ~10x under jitter; FL ~3x.
    FL's per-round chunking is robust to link impairment.

This module emits THREE separate figures so each can be cited
individually in the paper:

    jittery_walltime.png    --- 4-bar wall-time comparison (log y)
    jittery_energy.png      --- 4-bar energy comparison (log y)
    jittery_per_trial.png   --- per-trial scatter, side-by-side panels
                                with auto-scaled per-arm y-axes so the
                                within-arm spread is visible

Runnable directly::

    python -m experiments.analysis.exp1_jittery_deep_dive \\
        --csv-primary results/exp1_chameleon_no1gb.csv \\
        --csv-jittery results/exp1_chameleon_jittery.csv \\
        --out-dir DeveloperDocs/figures/exp1_jittery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis.exp1 import load_trials, per_row_energy
from experiments.calibration import load_calibration


def _filter_cell(rows, Dpd: str, alpha: float, R: int):
    return [r for r in rows
            if r.is_ok and r.Dpd == Dpd and r.alpha == alpha and r.R == R]


def _stats_by_arm(rows):
    out = {}
    for arm in ("FL", "Centralized"):
        ts = [r.Tproc_s for r in rows if r.arm == arm]
        ps = [r.Pcomplete for r in rows if r.arm == arm]
        if ts:
            out[arm] = (
                float(np.mean(ts)),
                float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0,
                ts,
                float(np.mean(ps)),
                len(ts),
            )
    return out


def _bar_figure(arms, means_p, sds_p, means_j, sds_j, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(arms))
    width = 0.38
    ax.bar(x - width / 2, means_p, width, yerr=sds_p, capsize=4,
            label="primary link (TBF only)", color="#4a8cd6")
    ax.bar(x + width / 2, means_j, width, yerr=sds_j, capsize=4,
            label="jittery link (TBF + netem)", color="#d65a3a")

    for xi, mp, mj in zip(x, means_p, means_j):
        infl = mj / mp if mp > 0 else float("nan")
        ax.annotate(
            f"{infl:.1f}x",
            xy=(xi + width / 2, mj), xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#a04020",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(min(means_p) * 0.4, max(means_j) * 2.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _per_trial_figure(fl_ts, ct_ts, out_path, cell_label):
    """Side-by-side scatter panels, each auto-scaled to its arm's spread.

    This honestly displays both clusters' internal variance --- which is
    invisible when the two are plotted on the same axis at very different
    magnitudes --- while preserving the magnitude gap via the y-axis
    labels. Annotations on each panel call out the mean/SD numerically
    and a centre band shows the inflation factor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              gridspec_kw={"width_ratios": [1, 1],
                                            "wspace": 0.25})

    fl_mean, fl_sd = float(np.mean(fl_ts)), float(np.std(fl_ts, ddof=1))
    ct_mean, ct_sd = float(np.mean(ct_ts)), float(np.std(ct_ts, ddof=1))

    rng = np.random.default_rng(42)
    fl_x = rng.normal(0.0, 0.04, size=len(fl_ts))
    ct_x = rng.normal(0.0, 0.04, size=len(ct_ts))

    # Left panel: FL
    ax_fl = axes[0]
    ax_fl.scatter(fl_x, fl_ts, alpha=0.7, s=70,
                    color="#2d7a2d", edgecolor="white", linewidth=0.5)
    ax_fl.axhline(fl_mean, color="black", linestyle="-", linewidth=1.5,
                    label=f"mean = {fl_mean:.2f} s")
    ax_fl.axhspan(fl_mean - fl_sd, fl_mean + fl_sd,
                    color="#2d7a2d", alpha=0.12,
                    label=f"+/- 1 SD = {fl_sd:.2f} s")
    fl_pad = max(fl_sd * 4.0, 0.5)
    ax_fl.set_ylim(fl_mean - fl_pad, fl_mean + fl_pad)
    ax_fl.set_xlim(-0.4, 0.4)
    ax_fl.set_xticks([])
    ax_fl.set_title(f"FL  (n={len(fl_ts)})", fontsize=12, color="#2d7a2d",
                     fontweight="bold")
    ax_fl.set_ylabel("T_proc (s)", fontsize=11)
    ax_fl.legend(loc="upper right", fontsize=9)
    ax_fl.grid(axis="y", alpha=0.3)

    # Right panel: Centralized
    ax_ct = axes[1]
    ax_ct.scatter(ct_x, ct_ts, alpha=0.7, s=70,
                    color="#c43a3a", edgecolor="white", linewidth=0.5)
    ax_ct.axhline(ct_mean, color="black", linestyle="-", linewidth=1.5,
                    label=f"mean = {ct_mean:.2f} s")
    ax_ct.axhspan(ct_mean - ct_sd, ct_mean + ct_sd,
                    color="#c43a3a", alpha=0.12,
                    label=f"+/- 1 SD = {ct_sd:.2f} s")
    ct_pad = max(ct_sd * 4.0, 5.0)
    ax_ct.set_ylim(ct_mean - ct_pad, ct_mean + ct_pad)
    ax_ct.set_xlim(-0.4, 0.4)
    ax_ct.set_xticks([])
    ax_ct.set_title(f"Centralized  (n={len(ct_ts)})",
                     fontsize=12, color="#c43a3a", fontweight="bold")
    ax_ct.set_ylabel("T_proc (s)", fontsize=11)
    ax_ct.legend(loc="upper right", fontsize=9)
    ax_ct.grid(axis="y", alpha=0.3)

    inflation = ct_mean / fl_mean if fl_mean > 0 else float("nan")
    fig.text(
        0.5, 0.02,
        f"Centralized mean is {inflation:.1f}x larger than FL mean    "
        f"(note the different y-axis ranges --- each panel zoomed to its "
        f"own cluster so the within-arm spread is visible)",
        ha="center", va="bottom", fontsize=10, style="italic",
        color="#444444",
    )
    fig.suptitle(
        f"Per-trial T_proc on jittery link --- {cell_label}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _walltime_per_trial_combined(
    arms, means_p, sds_p, means_j, sds_j,
    fl_ts, ct_ts, cell_label, out_path,
):
    """Combined wall-time bars + per-trial scatter into one 1x3 panel.

    Panel A: 4-bar log-scale wall-time comparison (FL/Cent x primary/jittery)
    Panel B: FL per-trial scatter (auto-scaled to FL cluster)
    Panel C: Centralized per-trial scatter (auto-scaled to Centralized cluster)
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5.2),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.28},
    )

    # -------- Panel A: 4-bar wall-time -----------------------------------
    ax = axes[0]
    x = np.arange(len(arms))
    width = 0.38
    ax.bar(x - width / 2, means_p, width, yerr=sds_p, capsize=4,
            label="primary link (TBF only)", color="#4a8cd6")
    ax.bar(x + width / 2, means_j, width, yerr=sds_j, capsize=4,
            label="jittery link (TBF + netem)", color="#d65a3a")

    for xi, mp, mj in zip(x, means_p, means_j):
        infl = mj / mp if mp > 0 else float("nan")
        ax.annotate(
            f"{infl:.1f}x",
            xy=(xi + width / 2, mj), xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#a04020",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=11)
    ax.set_ylabel("T_proc (s, log scale)", fontsize=11)
    ax.set_title(
        f"Wall-time comparison\n"
        f"jitter inflates Centralized {means_j[1]/means_p[1]:.1f}x  "
        f"vs FL {means_j[0]/means_p[0]:.1f}x",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(min(means_p) * 0.4, max(means_j) * 2.2)

    # -------- Panel B: FL per-trial --------------------------------------
    fl_mean, fl_sd = float(np.mean(fl_ts)), float(np.std(fl_ts, ddof=1))
    ct_mean, ct_sd = float(np.mean(ct_ts)), float(np.std(ct_ts, ddof=1))

    rng = np.random.default_rng(42)
    fl_x = rng.normal(0.0, 0.04, size=len(fl_ts))
    ct_x = rng.normal(0.0, 0.04, size=len(ct_ts))

    ax_fl = axes[1]
    ax_fl.scatter(fl_x, fl_ts, alpha=0.7, s=70,
                    color="#2d7a2d", edgecolor="white", linewidth=0.5)
    ax_fl.axhline(fl_mean, color="black", linestyle="-", linewidth=1.5,
                    label=f"mean = {fl_mean:.2f} s")
    ax_fl.axhspan(fl_mean - fl_sd, fl_mean + fl_sd,
                    color="#2d7a2d", alpha=0.12,
                    label=f"+/- 1 SD = {fl_sd:.2f} s")
    fl_pad = max(fl_sd * 4.0, 0.5)
    ax_fl.set_ylim(fl_mean - fl_pad, fl_mean + fl_pad)
    ax_fl.set_xlim(-0.4, 0.4)
    ax_fl.set_xticks([])
    ax_fl.set_title(f"FL per-trial (jittery)\nn={len(fl_ts)}",
                     fontsize=11, color="#2d7a2d", fontweight="bold")
    ax_fl.set_ylabel("T_proc (s)", fontsize=11)
    ax_fl.legend(loc="upper right", fontsize=9)
    ax_fl.grid(axis="y", alpha=0.3)

    # -------- Panel C: Centralized per-trial -----------------------------
    ax_ct = axes[2]
    ax_ct.scatter(ct_x, ct_ts, alpha=0.7, s=70,
                    color="#c43a3a", edgecolor="white", linewidth=0.5)
    ax_ct.axhline(ct_mean, color="black", linestyle="-", linewidth=1.5,
                    label=f"mean = {ct_mean:.2f} s")
    ax_ct.axhspan(ct_mean - ct_sd, ct_mean + ct_sd,
                    color="#c43a3a", alpha=0.12,
                    label=f"+/- 1 SD = {ct_sd:.2f} s")
    ct_pad = max(ct_sd * 4.0, 5.0)
    ax_ct.set_ylim(ct_mean - ct_pad, ct_mean + ct_pad)
    ax_ct.set_xlim(-0.4, 0.4)
    ax_ct.set_xticks([])
    ax_ct.set_title(f"Centralized per-trial (jittery)\nn={len(ct_ts)}",
                     fontsize=11, color="#c43a3a", fontweight="bold")
    ax_ct.set_ylabel("T_proc (s)", fontsize=11)
    ax_ct.legend(loc="upper right", fontsize=9)
    ax_ct.grid(axis="y", alpha=0.3)

    inflation = ct_mean / fl_mean if fl_mean > 0 else float("nan")
    fig.suptitle(
        f"Wall-time aggregate vs per-trial detail --- {cell_label}    "
        f"(per-trial panels zoomed to each arm's cluster; "
        f"Centralized mean is {inflation:.1f}x larger than FL mean)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.analysis.exp1_jittery_deep_dive")
    ap.add_argument("--csv-primary", required=True, type=Path)
    ap.add_argument("--csv-jittery", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--calibration", default=None, type=Path)
    ap.add_argument("--Dpd", default="100MB")
    ap.add_argument("--alpha", default=1.0, type=float)
    ap.add_argument("--R", default=20, type=int)
    args = ap.parse_args(argv)

    cal = load_calibration(args.calibration).exp1
    rows_p = _filter_cell(load_trials(args.csv_primary),
                            args.Dpd, args.alpha, args.R)
    rows_j = _filter_cell(load_trials(args.csv_jittery),
                            args.Dpd, args.alpha, args.R)
    if not (rows_p and rows_j):
        print("ERROR: cell missing from one or both CSVs.", file=sys.stderr)
        return 1

    sp = _stats_by_arm(rows_p)
    sj = _stats_by_arm(rows_j)

    e_p = per_row_energy(rows_p, cal)
    e_j = per_row_energy(rows_j, cal)
    energy_mean_p = {arm: float(np.mean([e.idle_J + e.tx_J for e in e_p
                                            if e.arm == arm]))
                      for arm in ("FL", "Centralized")}
    energy_mean_j = {arm: float(np.mean([e.idle_J + e.tx_J for e in e_j
                                            if e.arm == arm]))
                      for arm in ("FL", "Centralized")}
    energy_sd_p = {arm: float(np.std([e.idle_J + e.tx_J for e in e_p
                                          if e.arm == arm], ddof=1))
                    for arm in ("FL", "Centralized")}
    energy_sd_j = {arm: float(np.std([e.idle_J + e.tx_J for e in e_j
                                          if e.arm == arm], ddof=1))
                    for arm in ("FL", "Centralized")}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = ["FL", "Centralized"]
    cell_label = (f"Dpd={args.Dpd}, alpha={args.alpha:g}, R={args.R}, "
                  f"n=20 paired trials per arm")

    _bar_figure(
        arms,
        [sp[a][0] for a in arms], [sp[a][1] for a in arms],
        [sj[a][0] for a in arms], [sj[a][1] for a in arms],
        ylabel="T_proc (s, log scale)",
        title=(f"Wall-time --- {cell_label}\n"
               f"jitter inflates Centralized {sj['Centralized'][0]/sp['Centralized'][0]:.1f}x  "
               f"vs FL {sj['FL'][0]/sp['FL'][0]:.1f}x"),
        out_path=args.out_dir / "jittery_walltime.png",
    )

    _bar_figure(
        arms,
        [energy_mean_p[a] for a in arms], [energy_sd_p[a] for a in arms],
        [energy_mean_j[a] for a in arms], [energy_sd_j[a] for a in arms],
        ylabel="Energy (J, log scale)",
        title=(f"Energy proxy --- {cell_label}\n"
               f"P_idle={cal.P_idle_W:g} W, eps_bit={cal.epsilon_bit_J_per_bit:.1g} J/bit "
               f"(idle term dominates --- inflation tracks wall-time)"),
        out_path=args.out_dir / "jittery_energy.png",
    )

    _per_trial_figure(
        sj["FL"][2], sj["Centralized"][2],
        out_path=args.out_dir / "jittery_per_trial.png",
        cell_label=cell_label,
    )

    _walltime_per_trial_combined(
        arms,
        [sp[a][0] for a in arms], [sp[a][1] for a in arms],
        [sj[a][0] for a in arms], [sj[a][1] for a in arms],
        sj["FL"][2], sj["Centralized"][2],
        cell_label=cell_label,
        out_path=args.out_dir / "jittery_walltime_per_trial.png",
    )

    print(f"\nNumerical summary (cell: {cell_label}):")
    for a in arms:
        print(f"  {a:>12} primary: Tproc={sp[a][0]:7.2f}+/-{sp[a][1]:5.2f} s  "
              f"Pcomplete={sp[a][3]:.2f}  energy={energy_mean_p[a]:7.1f} J")
        print(f"  {a:>12} jittery: Tproc={sj[a][0]:7.2f}+/-{sj[a][1]:5.2f} s  "
              f"Pcomplete={sj[a][3]:.2f}  energy={energy_mean_j[a]:7.1f} J")
        print(f"  {a:>12} INFL   : Tproc {sj[a][0]/sp[a][0]:.2f}x   "
              f"energy {energy_mean_j[a]/energy_mean_p[a]:.2f}x")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
