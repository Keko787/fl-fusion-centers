"""Build the combined Experiment-1 calibration-sensitivity figure.

Replaces the two per-condition ``sensitivity_grid.png`` figures with a
single 2 x 5 grid where:

* Rows = link condition  (primary / jittery)
* Cols = calibration variant  (P_idle x0.5, baseline, P_idle x1.5,
                                eps_bit x0.5, eps_bit x1.5)

The y-axis is shared *within* each row (because primary and jittery
have very different absolute scales --- jittery T_proc is ~10x larger
under the netem overlay). Sharing the y-axis across columns makes the
rescaling visible; not sharing across rows keeps each row legible.

Runnable directly::

    python -m experiments.analysis.exp1_sensitivity_combined \\
        --csv-primary results/exp1_chameleon_no1gb.csv \\
        --csv-jittery results/exp1_chameleon_jittery.csv \\
        --out DeveloperDocs/figures/exp1/sensitivity_combined.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis.exp1 import _parse_size, load_trials, per_row_energy
from experiments.calibration import load_calibration


def _means(rows, cal):
    """Return (Dpd_order, arm_order, idle[arm,Dpd], tx[arm,Dpd]) at alpha=1.0."""
    e_rows = per_row_energy([r for r in rows if r.alpha == 1.0], cal)
    arms = sorted({e.arm for e in e_rows})
    Dpd_order = sorted({e.Dpd for e in e_rows}, key=lambda s: _parse_size(s))
    idle = np.zeros((len(arms), len(Dpd_order)))
    tx = np.zeros_like(idle)
    for i, arm in enumerate(arms):
        for j, d in enumerate(Dpd_order):
            cell = [e for e in e_rows if e.arm == arm and e.Dpd == d]
            if cell:
                idle[i, j] = np.mean([c.idle_J for c in cell])
                tx[i, j] = np.mean([c.tx_J for c in cell])
    return Dpd_order, arms, idle, tx


def _plot_panel(ax, Dpd_order, arms, idle, tx, ylim, show_xticks=True):
    x = np.arange(len(Dpd_order))
    width = 0.35
    handles = []
    for i, arm in enumerate(arms):
        offset = (i - 0.5) * width
        h_idle = ax.bar(x + offset, idle[i], width, label=f"{arm} idle")
        h_tx = ax.bar(x + offset, tx[i], width, bottom=idle[i],
                       label=f"{arm} tx")
        handles.extend([h_idle, h_tx])
    ax.set_xticks(x)
    if show_xticks:
        ax.set_xticklabels(Dpd_order, fontsize=9)
    else:
        ax.set_xticklabels([])
    ax.set_ylim(0, ylim)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    return handles


def _build_row(rows, cals, panel_titles):
    """Compute panel data for one row and return ((data...), ymax)."""
    panel_data = []
    ymax = 0.0
    for cal in cals:
        Dpd_order, arms, idle, tx = _means(rows, cal)
        panel_data.append((Dpd_order, arms, idle, tx))
        ymax = max(ymax, float(np.max(idle + tx)))
    return panel_data, ymax * 1.10


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.analysis.exp1_sensitivity_combined")
    ap.add_argument("--csv-primary", required=True, type=Path)
    ap.add_argument("--csv-jittery", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--baseline-toml", default=None, type=Path)
    ap.add_argument("--variants-dir",
                     default=Path("experiments/calibration_sensitivity"),
                     type=Path)
    args = ap.parse_args(argv)

    rows_primary = load_trials(args.csv_primary)
    rows_jittery = load_trials(args.csv_jittery)

    base = load_calibration(args.baseline_toml).exp1
    p_lo = load_calibration(args.variants_dir / "p_idle_low.toml").exp1
    p_hi = load_calibration(args.variants_dir / "p_idle_high.toml").exp1
    e_lo = load_calibration(args.variants_dir / "eps_low.toml").exp1
    e_hi = load_calibration(args.variants_dir / "eps_high.toml").exp1

    cals = [p_lo, base, p_hi, e_lo, e_hi]
    col_titles = [
        f"P_idle x0.5\n({p_lo.P_idle_W:g} W)",
        f"baseline\n({base.P_idle_W:g} W, eps={base.epsilon_bit_J_per_bit:.2g})",
        f"P_idle x1.5\n({p_hi.P_idle_W:g} W)",
        f"eps_bit x0.5\n({e_lo.epsilon_bit_J_per_bit:.2g} J/bit)",
        f"eps_bit x1.5\n({e_hi.epsilon_bit_J_per_bit:.2g} J/bit)",
    ]

    primary_panels, ymax_primary = _build_row(rows_primary, cals, col_titles)
    jittery_panels, ymax_jittery = _build_row(rows_jittery, cals, col_titles)

    fig, axes = plt.subplots(
        nrows=2, ncols=5,
        figsize=(16, 7.5),
        sharey="row",
    )

    legend_handles = None
    legend_labels = None
    for col, (data, title) in enumerate(zip(primary_panels, col_titles)):
        Dpd_order, arms, idle, tx = data
        h = _plot_panel(axes[0, col], Dpd_order, arms, idle, tx,
                         ymax_primary, show_xticks=False)
        axes[0, col].set_title(title, fontsize=10)
        if col == 0 and legend_handles is None:
            legend_handles, legend_labels = (
                axes[0, col].get_legend_handles_labels()
            )

    for col, data in enumerate(jittery_panels):
        Dpd_order, arms, idle, tx = data
        _plot_panel(axes[1, col], Dpd_order, arms, idle, tx,
                     ymax_jittery, show_xticks=True)

    # Row labels on the leftmost axes.
    axes[0, 0].set_ylabel("Energy (J)\nprimary link (TBF only)", fontsize=10)
    axes[1, 0].set_ylabel("Energy (J)\njittery link (TBF + netem)", fontsize=10)

    # Single shared legend below the bottom row.
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=4, fontsize=10,
        title="Energy decomposition (stacked: idle + tx)",
        title_fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Calibration sensitivity sweep --- primary vs jittery link "
        "(energy proxy at alpha=1.0; y-axis shared within each row)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
