"""Build the EX-1 calibration-sensitivity comparison grid.

Reads the trial CSV once, then computes the energy proxy under the
baseline calibration plus the four ±50 % sensitivity variants (P_idle
low/high and epsilon_bit low/high). Lays the five resulting energy
stacked-bar panels into one figure with a shared y-axis so the
rescaling is visually obvious at a glance.

Runnable directly::

    python -m experiments.analysis.exp1_sensitivity_panel \\
        --csv results/exp1_chameleon_no1gb.csv \\
        --out DeveloperDocs/figures/exp1/sensitivity_grid.png

The five-panel layout is 2 rows x 3 columns with the bottom-right cell
blank so the figure fits on a half-page in two-column papers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis.exp1 import _parse_size, load_trials, per_row_energy
from experiments.calibration import Exp1Calibration, load_calibration


def _means(rows, cal):
    """Return (Dpd_order, arm_order, idle[arm,Dpd], tx[arm,Dpd]) at α=1.0."""
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


def _plot_panel(ax, Dpd_order, arms, idle, tx, title, ylim):
    x = np.arange(len(Dpd_order))
    width = 0.35
    for i, arm in enumerate(arms):
        offset = (i - 0.5) * width
        ax.bar(x + offset, idle[i], width, label=f"{arm} idle")
        ax.bar(x + offset, tx[i], width, bottom=idle[i], label=f"{arm} tx")
    ax.set_xticks(x)
    ax.set_xticklabels(Dpd_order, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, ylim)
    ax.tick_params(axis="y", labelsize=8)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.analysis.exp1_sensitivity_panel")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Per-trial CSV (e.g. results/exp1_chameleon_no1gb.csv)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output PNG path.")
    ap.add_argument("--baseline-toml", default=None, type=Path,
                    help="Override baseline calibration TOML (else default).")
    ap.add_argument("--variants-dir",
                    default=Path("experiments/calibration_sensitivity"),
                    type=Path,
                    help="Dir containing p_idle_low/high and eps_low/high TOMLs.")
    args = ap.parse_args(argv)

    rows = load_trials(args.csv)

    # Five calibrations: baseline + four variants.
    base = load_calibration(args.baseline_toml).exp1
    p_lo = load_calibration(args.variants_dir / "p_idle_low.toml").exp1
    p_hi = load_calibration(args.variants_dir / "p_idle_high.toml").exp1
    e_lo = load_calibration(args.variants_dir / "eps_low.toml").exp1
    e_hi = load_calibration(args.variants_dir / "eps_high.toml").exp1

    panels = [
        (f"P_idle x0.5  ({p_lo.P_idle_W:g} W)", p_lo),
        (f"baseline  (P_idle={base.P_idle_W:g} W, eps={base.epsilon_bit_J_per_bit:.2g})", base),
        (f"P_idle x1.5  ({p_hi.P_idle_W:g} W)", p_hi),
        (f"eps_bit x0.5  ({e_lo.epsilon_bit_J_per_bit:.2g} J/bit)", e_lo),
        (f"eps_bit x1.5  ({e_hi.epsilon_bit_J_per_bit:.2g} J/bit)", e_hi),
    ]

    # First pass: compute all data and find the global ylim for shared axes.
    all_data = []
    ymax = 0.0
    for title, cal in panels:
        Dpd_order, arms, idle, tx = _means(rows, cal)
        all_data.append((title, Dpd_order, arms, idle, tx))
        ymax = max(ymax, float(np.max(idle + tx)))
    ymax *= 1.10  # 10 % headroom

    # 2 x 3 grid; bottom-right empty.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    flat = axes.ravel()
    for i, (title, Dpd_order, arms, idle, tx) in enumerate(all_data):
        _plot_panel(flat[i], Dpd_order, arms, idle, tx, title, ymax)
    flat[5].axis("off")  # blank bottom-right

    # One shared legend in the blank cell.
    handles, labels = flat[0].get_legend_handles_labels()
    flat[5].legend(handles, labels, loc="center", fontsize=10,
                    title="Energy decomposition", title_fontsize=10)

    # Common y-label on the leftmost column only.
    for row in (0, 1):
        axes[row, 0].set_ylabel("Energy (J)")

    fig.suptitle(
        "Calibration sensitivity sweep (energy proxy at alpha=1.0, shared y-axis)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
