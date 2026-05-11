"""Bar chart of per-client class distribution from ``partition_stats.json``.

Phase E.4 — outline §6.7 figure showing the non-IID character of the
geographic partition.

Usage:
    python -m Analysis.CommunitiesCrime.plot_per_client_distribution \\
        --run_dir results/commcrime_run_<ts> \\
        --output  results/commcrime_run_<ts>/per_client_distribution.png

Output formats: extension drives format (``.png``, ``.pdf``, ``.svg``,
``.eps`` all supported via matplotlib). For papers prefer
``--style paper`` (300 DPI, no top/right spines, larger fonts).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from Analysis.CommunitiesCrime.log_parser import parse_partition_stats
from Analysis.CommunitiesCrime.plot_style import apply_style, derive_savefig_kwargs


_CLASS_LABELS = {"0": "violent", "1": "property", "2": "other"}


def plot(run_dir: str | Path,
         output: str | Path | None = None,
         style: str = "default",
         show: bool = False,
         close: bool = True) -> Tuple[Path, Figure]:
    """Render the per-client class-distribution bar chart.

    Args:
        run_dir: directory containing ``partition_stats.json``.
        output: output path (extension determines format).
            Defaults to ``<run_dir>/per_client_distribution.png``.
        style: ``"default"`` or ``"paper"`` — see :mod:`plot_style`.
        show: open an interactive window after rendering.
        close: close the figure with ``plt.close(fig)`` after saving
            (default True for CLI use; notebooks pass ``False`` and
            re-render via the returned Figure).

    Returns ``(output_path, figure)``.
    """
    run_dir = Path(run_dir)
    stats = parse_partition_stats(run_dir / "partition_stats.json")
    clients = stats.get("clients", {})

    if not clients:
        raise ValueError(
            f"No client entries in {run_dir / 'partition_stats.json'}"
        )

    client_ids = sorted(clients.keys(), key=int)
    classes = sorted({c for cid in client_ids
                      for c in clients[cid]["train"].get("class_distribution", {})})

    counts = np.zeros((len(classes), len(client_ids)), dtype=int)
    for j, cid in enumerate(client_ids):
        dist = clients[cid]["train"].get("class_distribution", {})
        for i, c in enumerate(classes):
            counts[i, j] = int(dist.get(c, 0))

    with apply_style(style):
        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(client_ids)), 5))
        bottom = np.zeros(len(client_ids), dtype=int)
        x = np.arange(len(client_ids))
        for i, c in enumerate(classes):
            ax.bar(x, counts[i], bottom=bottom,
                   label=_CLASS_LABELS.get(c, f"class {c}"))
            bottom = bottom + counts[i]

        ax.set_xlabel("Client ID")
        ax.set_ylabel("Train samples")
        ax.set_title(
            f"Per-client class distribution "
            f"(strategy={stats.get('strategy')}, N={stats.get('num_clients')})"
        )
        ax.set_xticks(x, client_ids)
        ax.legend(title="threat_class")
        fig.tight_layout()

        out_path = Path(output) if output else run_dir / "per_client_distribution.png"
        fig.savefig(out_path, **derive_savefig_kwargs(out_path))
        if show:
            plt.show()
        if close:
            plt.close(fig)
    return out_path, fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="Run directory containing partition_stats.json.")
    parser.add_argument("--output", default=None,
                        help="Output path; extension picks format (default: <run_dir>/per_client_distribution.png).")
    parser.add_argument("--style", default="default", choices=["default", "paper"],
                        help="Plot style preset.")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive window after rendering.")
    args = parser.parse_args()
    path, _ = plot(args.run_dir, args.output, style=args.style, show=args.show)
    print(f"=== Wrote {path} ===")


if __name__ == "__main__":
    main()
