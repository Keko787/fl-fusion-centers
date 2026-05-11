"""Scaling figure — best smoothed macro-F1 + rounds-to-convergence vs N.

Phase E.4 — outline §7.4 row 6 (Scaling: N=3, 5, 10). Reads one
``server_evaluation.log`` per N and produces a twin-axis figure with
best-smoothed macro-F1 (bars) and rounds-to-convergence (line).

Bar height uses a rolling-mean smoothed series (Phase E review #6) so
the value reflects what the model converged to, not last-round noise.
Convergence is the first round whose smoothed metric is within a
fraction of the smoothed max (review #5).

Output formats: extension drives format (``.png``, ``.pdf``, ``.svg``).
For papers prefer ``--style paper`` (300 DPI, larger fonts).

Usage:
    python -m Analysis.CommunitiesCrime.plot_scaling_n_clients \\
        --runs 3=run_n3 5=run_n5 10=run_n10 \\
        --output scaling.pdf --style paper
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from Analysis.CommunitiesCrime.log_parser import parse_server_log
from Analysis.CommunitiesCrime.plot_style import apply_style, derive_savefig_kwargs


def _smoothed_series(df: pd.DataFrame, metric: str,
                      smooth_window: int) -> pd.Series:
    """Rolling-mean smoothed copy of ``df[metric]`` keeping the index aligned.

    Uses ``min_periods=1`` so early rounds still get a value (running
    mean over however many rounds exist so far) — keeps the series the
    same length as ``df`` for index-aligned operations.
    """
    return df[metric].rolling(window=smooth_window, min_periods=1).mean()


def _best_smoothed(df: pd.DataFrame, metric: str,
                    smooth_window: int = 5) -> float:
    """Best value of the smoothed metric — robust to a single overshoot
    that the last round happens to capture."""
    if df.empty or metric not in df.columns:
        return 0.0
    return float(_smoothed_series(df, metric, smooth_window).max())


def _rounds_to_convergence(df: pd.DataFrame, metric: str = "threat_macro_f1",
                            smooth_window: int = 5,
                            tolerance_frac: float = 0.05) -> int:
    """First round whose smoothed metric is within ``tolerance_frac`` of
    the smoothed maximum.

    Improvements over the original heuristic (Phase E review #5):
      * **Smoothed** — rolling mean over ``smooth_window`` rounds, so a
        single noisy round can't be mistaken for convergence.
      * **Anchored to max-of-smoothed**, not last-round value — overshoot-
        and-decline runs no longer mark early rounds as "converged"
        just because they happen to be near the declined final value.
      * **Tolerance is a fraction of the dynamic range** (max - min of
        smoothed), not an absolute 0.01. Degenerate flat runs fall back
        to an absolute 1e-6 threshold.
    """
    if df.empty or metric not in df.columns:
        return 0
    smoothed = _smoothed_series(df, metric, smooth_window)
    max_val = float(smoothed.max())
    min_val = float(smoothed.min())
    span = max_val - min_val
    abs_tol = (tolerance_frac * span) if span > 1e-9 else 1e-6
    within = smoothed >= (max_val - abs_tol)
    if not within.any():
        return int(df["round"].iloc[-1])
    first_idx = within.idxmax()
    return int(df.loc[first_idx, "round"])


def plot(runs: dict[int, str | Path],
         output: str | Path | None = None,
         metric: str = "threat_macro_f1",
         smooth_window: int = 5,
         tolerance_frac: float = 0.05,
         style: str = "default",
         show: bool = False,
         close: bool = True) -> Tuple[Path, Figure]:
    """Render the scaling chart.

    Args:
        runs: ``{N: run_dir}`` mapping; each run_dir must contain
            ``server_evaluation.log``.
        output: output path; extension picks format. Defaults to
            ``./scaling_n_clients.png``.
        metric: which metric to use for the bar height. The bar shows
            the **best smoothed value** over the run, not the last-round
            value (Phase E review #6) — robust to training overshoot.
        smooth_window: rolling-mean window for the smoothed curve used
            both for the bar height and for the convergence calculation.
        tolerance_frac: fraction of the smoothed dynamic range used to
            declare convergence (Phase E review #5).
        style: ``"default"`` or ``"paper"``.
        close: close the figure after saving (default True for CLI).

    Returns ``(output_path, figure)``.
    """
    if not runs:
        raise ValueError("No runs provided.")

    ns = sorted(runs.keys())
    final_metric: list[float] = []
    rounds_to_conv: list[int] = []
    for n in ns:
        df = parse_server_log(Path(runs[n]) / "server_evaluation.log")
        final_metric.append(_best_smoothed(df, metric, smooth_window))
        rounds_to_conv.append(_rounds_to_convergence(
            df, metric, smooth_window, tolerance_frac,
        ))

    out_path = Path(output) if output else Path("scaling_n_clients.png")

    with apply_style(style):
        x = np.arange(len(ns))
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(x, final_metric, color="steelblue", alpha=0.7,
                 label=f"Best smoothed {metric}")
        ax1.set_xlabel("Number of clients (N)")
        ax1.set_ylabel(f"Best smoothed {metric}", color="steelblue")
        ax1.set_xticks(x, [str(n) for n in ns])
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_ylim(0, max(final_metric) * 1.2 if final_metric else 1.0)

        ax2 = ax1.twinx()
        ax2.plot(x, rounds_to_conv, "o-", color="firebrick",
                  label="Rounds to convergence")
        ax2.set_ylabel("Rounds to convergence", color="firebrick")
        ax2.tick_params(axis="y", labelcolor="firebrick")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax1.set_title(f"Scaling — best smoothed {metric} + rounds to convergence "
                       f"(smooth={smooth_window}, tol={tolerance_frac:.0%})")
        fig.tight_layout()

        fig.savefig(out_path, **derive_savefig_kwargs(out_path))
        if show:
            plt.show()
        if close:
            plt.close(fig)
    return out_path, fig


def _parse_runs(items: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"--runs entries must be of the form N=path; got {item!r}"
            )
        n_str, path = item.split("=", 1)
        out[int(n_str)] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more 'N=run_dir' pairs (e.g. '5=results/run_n5').")
    parser.add_argument("--metric", default="threat_macro_f1")
    parser.add_argument("--output", default=None,
                        help="Output path; extension picks format (default: ./scaling_n_clients.png).")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Rolling-mean window for the smoothed curve (default: 5).")
    parser.add_argument("--tolerance-frac", type=float, default=0.05,
                        help="Fraction of smoothed dynamic range for convergence (default: 0.05).")
    parser.add_argument("--style", default="default", choices=["default", "paper"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    path, _ = plot(_parse_runs(args.runs), args.output,
                    metric=args.metric,
                    smooth_window=args.smooth_window,
                    tolerance_frac=args.tolerance_frac,
                    style=args.style, show=args.show)
    print(f"=== Wrote {path} ===")


if __name__ == "__main__":
    main()
