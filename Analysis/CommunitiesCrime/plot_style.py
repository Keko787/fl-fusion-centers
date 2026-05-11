"""Shared plot styling for the fusion-centers figures.

Phase E review #15 — bumps the default DPI to 300, sets a small,
predictable color palette, and ensures tight layout. ``--style paper``
on the CLI activates this; the matplotlib default style remains
available via ``--style default`` for ad-hoc exploration.

All three plot scripts use :func:`apply_style` at the top of
``plot(...)``; it is idempotent and process-local (no rcParams leak
across pytest runs because matplotlib's rc context manager is used).
"""
from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt


_PAPER_RC = {
    "figure.dpi": 100,           # screen-side; savefig DPI is separate
    "savefig.dpi": 300,          # journal-quality
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
}


@contextmanager
def apply_style(style: str = "default"):
    """Context manager that applies one of the named styles for the
    duration of a single plot call.

    Styles:
      * ``"default"`` — matplotlib defaults (current PNG output at 150 DPI).
      * ``"paper"``   — journal-grade: 300 DPI savefig, tight bbox, no
        top/right spines, larger fonts, frameless legend.
    """
    if style == "paper":
        with mpl.rc_context(rc=_PAPER_RC):
            yield
    elif style == "default":
        yield
    else:
        raise ValueError(f"Unknown plot style: {style!r}")


def derive_savefig_kwargs(output_path) -> dict:
    """Build ``fig.savefig`` kwargs from the output filename extension.

    matplotlib auto-detects format from the extension, so this helper
    is mostly about adding dpi when the extension is raster-only.
    Vector formats (``.pdf``, ``.svg``, ``.eps``) ignore the dpi kwarg.
    """
    from pathlib import Path
    ext = Path(output_path).suffix.lower().lstrip(".")
    if not ext:
        # No extension → default to PNG.
        return {"format": "png", "dpi": 200}
    if ext in {"pdf", "svg", "eps", "ps"}:
        # Vector — dpi is irrelevant.
        return {"format": ext}
    return {"format": ext, "dpi": 200}
