"""Headline figure — macro-F1 over rounds for FedAvg vs FedProx vs Centralized.

Phase E.4 — outline §7.9 primary plot. Reads multiple ``server_evaluation.log``
files and overlays them, optionally with a horizontal reference line for
the centralized baseline macro-F1.

Output formats: extension drives format (``.png``, ``.pdf``, ``.svg``,
``.eps`` all supported via matplotlib). For papers prefer
``--style paper`` (300 DPI, no top/right spines, larger fonts).

Usage:
    python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \\
        --runs label=path label=path ... \\
        --centralized-baseline 0.62 \\
        --output headline.pdf \\
        --style paper

The --runs argument is a list of ``label=run_dir`` pairs; each run_dir
must contain ``server_evaluation.log``. The label is what appears in
the figure legend.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from Analysis.CommunitiesCrime.log_parser import collect_server_logs
from Analysis.CommunitiesCrime.plot_style import apply_style, derive_savefig_kwargs


def plot(runs: dict[str, str | Path],
         output: str | Path | None = None,
         centralized_baseline: Optional[float] = None,
         metric: str = "threat_macro_f1",
         style: str = "default",
         show: bool = False,
         close: bool = True) -> Tuple[Path, Figure]:
    """Render the centralized-vs-federated convergence curve.

    Args:
        runs: ``{label: run_dir}`` mapping; each ``run_dir`` must contain
            ``server_evaluation.log``.
        output: output path; extension picks format. Defaults to
            ``./centralized_vs_federated.png`` in CWD.
        centralized_baseline: optional scalar; drawn as a horizontal
            dashed reference line.
        metric: which y-axis metric to plot. Defaults to ``threat_macro_f1``.
        style: ``"default"`` or ``"paper"``.
        close: close the figure after saving (default True for CLI).

    Returns ``(output_path, figure)``.
    """
    if not runs:
        raise ValueError("No runs provided.")

    df = collect_server_logs(list(runs.values()), list(runs.keys()))
    out_path = Path(output) if output else Path("centralized_vs_federated.png")

    with apply_style(style):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, sub in df.groupby("label"):
            if metric not in sub.columns:
                continue
            ax.plot(sub["round"], sub[metric], label=label, marker="o", markersize=3)

        if centralized_baseline is not None:
            ax.axhline(centralized_baseline, linestyle="--", color="black",
                       alpha=0.6, label=f"Centralized ({centralized_baseline:.3f})")

        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.set_title(f"Federated convergence — {metric}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.savefig(out_path, **derive_savefig_kwargs(out_path))
        if show:
            plt.show()
        if close:
            plt.close(fig)
    return out_path, fig


def _parse_runs(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"--runs entries must be of the form label=path; got {item!r}"
            )
        label, path = item.split("=", 1)
        out[label] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="One or more 'label=run_dir' pairs (e.g. 'FedAvg-IID=results/run_a').",
    )
    parser.add_argument("--centralized-baseline", type=float, default=None,
                        help="Horizontal reference line at this macro-F1 value.")
    parser.add_argument("--metric", default="threat_macro_f1",
                        help="Y-axis metric (default: threat_macro_f1).")
    parser.add_argument("--output", default=None,
                        help="Output path; extension picks format (default: ./centralized_vs_federated.png).")
    parser.add_argument("--style", default="default", choices=["default", "paper"],
                        help="Plot style preset.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    path, _ = plot(_parse_runs(args.runs), args.output,
                    centralized_baseline=args.centralized_baseline,
                    metric=args.metric, style=args.style, show=args.show)
    print(f"=== Wrote {path} ===")


if __name__ == "__main__":
    main()
