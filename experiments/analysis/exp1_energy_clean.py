"""Honest 3-bar energy figure for Experiment 1.

The default ``exp1_energy_stacked_bar.png`` shows four bars (Centralized
x {10MB, 100MB} and FL x {10MB, 100MB}). But FL energy is essentially
invariant in Dpd --- the two FL bars differ by < 1 % --- so plotting
both implies a Dpd dependence FL does not have.

This module emits a three-bar figure instead:

    FL  |  Centralized 10MB  |  Centralized 100MB

with FL pooled across Dpd (since invariant) and Centralized broken out
by Dpd (since the data shipping is what scales). Error bars show the
trial-level standard deviation, so a reader sees the FL bar's spread
honestly --- it is wide because the FL bar pools R={5,20,50} runs,
which span ~20 to ~210 J at P_idle = 5 W.

Runnable directly::

    python -m experiments.analysis.exp1_energy_clean \\
        --csv results/exp1_chameleon_no1gb.csv \\
        --out DeveloperDocs/figures/exp1/exp1_energy_clean.png
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


def _stack(samples_idle, samples_tx):
    """Mean and std of idle/tx given lists of per-trial values."""
    return (
        float(np.mean(samples_idle)),
        float(np.mean(samples_tx)),
        float(np.std(samples_idle, ddof=1)) if len(samples_idle) > 1 else 0.0,
        float(np.std(samples_tx, ddof=1)) if len(samples_tx) > 1 else 0.0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.analysis.exp1_energy_clean")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--calibration", default=None, type=Path,
                    help="Override calibration TOML; defaults to verified.")
    ap.add_argument("--alpha", default=1.0, type=float,
                    help="Filter trials to this alpha value (default 1.0).")
    args = ap.parse_args(argv)

    cal = load_calibration(args.calibration).exp1
    rows = load_trials(args.csv)
    e_rows = per_row_energy([r for r in rows if r.alpha == args.alpha], cal)

    fl_rows = [e for e in e_rows if e.arm == "FL"]
    cent_10 = [e for e in e_rows if e.arm == "Centralized" and e.Dpd == "10MB"]
    cent_100 = [e for e in e_rows if e.arm == "Centralized" and e.Dpd == "100MB"]

    if not (fl_rows and cent_10 and cent_100):
        print("ERROR: missing one of FL / Centralized 10MB / Centralized 100MB",
              file=sys.stderr)
        return 1

    fl_idle, fl_tx, fl_idle_sd, fl_tx_sd = _stack(
        [e.idle_J for e in fl_rows], [e.tx_J for e in fl_rows]
    )
    c10_idle, c10_tx, c10_idle_sd, c10_tx_sd = _stack(
        [e.idle_J for e in cent_10], [e.tx_J for e in cent_10]
    )
    c100_idle, c100_tx, c100_idle_sd, c100_tx_sd = _stack(
        [e.idle_J for e in cent_100], [e.tx_J for e in cent_100]
    )

    labels = [
        f"FL\n(pooled over Dpd, R; n={len(fl_rows)})",
        f"Centralized 10MB\n(n={len(cent_10)})",
        f"Centralized 100MB\n(n={len(cent_100)})",
    ]
    idle_vals = [fl_idle, c10_idle, c100_idle]
    tx_vals = [fl_tx, c10_tx, c100_tx]
    total_sd = [
        float(np.sqrt(fl_idle_sd ** 2 + fl_tx_sd ** 2)),
        float(np.sqrt(c10_idle_sd ** 2 + c10_tx_sd ** 2)),
        float(np.sqrt(c100_idle_sd ** 2 + c100_tx_sd ** 2)),
    ]
    totals = [i + t for i, t in zip(idle_vals, tx_vals)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.55
    colors = {"idle": "#3b82c4", "tx": "#e08428"}

    bars_idle = ax.bar(x, idle_vals, width, label="idle (T_proc * P_idle)",
                        color=colors["idle"])
    bars_tx = ax.bar(x, tx_vals, width, bottom=idle_vals,
                      label="tx (B_pw * 8 * eps_bit)",
                      color=colors["tx"])

    ax.errorbar(
        x, totals, yerr=total_sd, fmt="none",
        ecolor="black", elinewidth=1.2, capsize=5,
    )

    for xi, total, sd in zip(x, totals, total_sd):
        ax.annotate(
            f"{total:.0f} J\n+/- {sd:.0f}",
            xy=(xi, total + sd), xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Energy (J)")
    ax.set_title(
        f"Energy proxy at alpha={args.alpha:g}  "
        f"(P_idle={cal.P_idle_W:g} W, eps_bit={cal.epsilon_bit_J_per_bit:.2g} J/bit)\n"
        f"FL pooled across Dpd (invariant by design); Centralized broken out by Dpd"
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(totals + total_sd) * 1.18)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    print("\nNumerical summary:")
    print(f"  FL                      : {fl_idle:.2f} J idle + {fl_tx:.4f} J tx "
          f"= {fl_idle + fl_tx:.2f} J  (n={len(fl_rows)}, sd={total_sd[0]:.2f})")
    print(f"  Centralized 10MB        : {c10_idle:.2f} J idle + {c10_tx:.4f} J tx "
          f"= {c10_idle + c10_tx:.2f} J  (n={len(cent_10)}, sd={total_sd[1]:.2f})")
    print(f"  Centralized 100MB       : {c100_idle:.2f} J idle + {c100_tx:.4f} J tx "
          f"= {c100_idle + c100_tx:.2f} J  (n={len(cent_100)}, sd={total_sd[2]:.2f})")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
