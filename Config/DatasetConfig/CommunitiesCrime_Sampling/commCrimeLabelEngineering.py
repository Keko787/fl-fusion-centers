"""Label engineering for UCI Communities and Crime (Phase A.2 / outline §6.4).

Two targets are derived once, up front, before federated partitioning so
every client shares the same label space:

* ``threat_class`` (int, 0/1/2) — argmax of per-row normalized rates over
  three crime families: violent, property, other.
* ``escalation_score`` (float in [0, 1]) — weighted severity index. Min-max
  scaled to [0, 1] across the dataset so the sigmoid head has a clean target.

Weights and family memberships are exposed as module-level constants so
they can be tuned without touching the engineering code. Document any
change in the methods section of the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Outline §6.4 crime-family taxonomy. Violent and property are the two
# UCR Part I groupings; arson is the canonical "other" residual.
VIOLENT_RATE_COLS: tuple[str, ...] = (
    "murdPerPop", "rapesPerPop", "robbbPerPop", "assaultPerPop",
)
PROPERTY_RATE_COLS: tuple[str, ...] = (
    "burglPerPop", "larcPerPop", "autoTheftPerPop",
)
OTHER_RATE_COLS: tuple[str, ...] = (
    "arsonsPerPop",
)

THREAT_CLASS_LABELS: tuple[str, ...] = ("violent", "property", "other")

# Escalation-score weights. Violent crimes weighted highest, property
# second, other lowest (outline §6.4). Weights are applied to the raw
# family rates (sum per family) and then min-max scaled.
DEFAULT_ESCALATION_WEIGHTS = {"violent": 0.6, "property": 0.3, "other": 0.1}


def _family_rate(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    """Sum of per-100k rates across one crime family."""
    present = [c for c in cols if c in df.columns]
    if not present:
        raise KeyError(f"No crime-rate columns from {cols} present in dataframe")
    return df[list(present)].sum(axis=1)


def derive_threat_class(df: pd.DataFrame) -> pd.Series:
    """Per-row dominant crime family (0=violent, 1=property, 2=other).

    Rows where all three family rates are zero (or NaN) get class 2
    ("other") as a deterministic fallback. ``clean()`` drops rows
    missing crime targets before this is called, so all-zero rows are
    rare but possible.
    """
    violent = _family_rate(df, VIOLENT_RATE_COLS)
    property_ = _family_rate(df, PROPERTY_RATE_COLS)
    other = _family_rate(df, OTHER_RATE_COLS)

    stacked = np.column_stack([violent.to_numpy(),
                                property_.to_numpy(),
                                other.to_numpy()])
    totals = stacked.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    normalized = stacked / safe_totals
    classes = normalized.argmax(axis=1)
    # All-zero rows: argmax returns 0; remap to 2 (other) so the fallback
    # is explicit and doesn't bias the violent class.
    classes = np.where(totals.squeeze() > 0, classes, 2)
    return pd.Series(classes.astype(np.int64), index=df.index, name="threat_class")


def derive_escalation_score(df: pd.DataFrame,
                             weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted severity index, min-max scaled to [0, 1]."""
    w = weights or DEFAULT_ESCALATION_WEIGHTS
    raw = (w["violent"] * _family_rate(df, VIOLENT_RATE_COLS)
           + w["property"] * _family_rate(df, PROPERTY_RATE_COLS)
           + w["other"] * _family_rate(df, OTHER_RATE_COLS))
    lo = float(raw.min())
    hi = float(raw.max())
    if hi - lo < 1e-12:
        # Degenerate dataset (all rows identical severity). Return 0.5
        # rather than failing — the BCE head still trains, just trivially.
        return pd.Series(np.full(len(raw), 0.5, dtype=np.float32),
                         index=df.index, name="escalation_score")
    scaled = (raw - lo) / (hi - lo)
    return pd.Series(scaled.astype(np.float32), index=df.index,
                     name="escalation_score")


def engineer_labels(df: pd.DataFrame,
                    weights: dict[str, float] | None = None,
                    drop_rate_columns: bool = True) -> pd.DataFrame:
    """Return a copy of ``df`` with ``threat_class`` and ``escalation_score`` added.

    When ``drop_rate_columns=True`` (default), the raw crime-rate columns
    are dropped from the returned frame so they can't accidentally leak
    into model features. Set to ``False`` only for analysis / sanity
    checks where you want to inspect the rates alongside the derived
    labels.
    """
    out = df.copy()
    out["threat_class"] = derive_threat_class(df)
    out["escalation_score"] = derive_escalation_score(df, weights)
    if drop_rate_columns:
        rate_cols = (VIOLENT_RATE_COLS + PROPERTY_RATE_COLS + OTHER_RATE_COLS)
        present = [c for c in rate_cols if c in out.columns]
        if present:
            out = out.drop(columns=present)
    return out
