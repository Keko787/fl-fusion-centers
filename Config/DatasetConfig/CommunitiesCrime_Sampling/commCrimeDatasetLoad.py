"""UCI Communities and Crime (Unnormalized) raw loader.

Phase A.1 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

Covers outline §6 phases 1–3: acquisition, schema audit, cleaning. Label
engineering (§6.4) lives in ``commCrimeLabelEngineering.py``; partitioning
(§6.7) lives in ``commCrimeFederatedPartition.py``.

The UCI archive URL is recorded as a default constant; pass an explicit
``url`` to override if the upstream layout changes. For offline tests
use ``make_synthetic_stub`` to generate a deterministic small CSV that
matches the real schema closely enough for end-to-end smoke runs.
"""
from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/"
    "CommViolPredUnnormalizedData.txt"
)
DEFAULT_NAMES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/"
    "communities%20and%20crime%20unnormalized.names"
)
DEFAULT_TARGET_DIR = Path.home() / "datasets" / "CommunitiesCrime"
RAW_DATA_FILENAME = "CommViolPredUnnormalizedData.txt"
RAW_NAMES_FILENAME = "communities_and_crime_unnormalized.names"

# Non-predictive identifier columns dropped during cleaning (outline §6.3).
# ``state`` is retained for partitioning and dropped later by the
# preprocessing step before feature scaling.
ID_COLUMNS: tuple[str, ...] = (
    "communityname", "countyCode", "communityCode", "fold",
)

# Sensitive demographic columns dropped when ``drop_sensitive=True``
# (the default per design doc §8.4, the Phase 0 locked decision).
#
# Taxonomy:
#   * Race/ethnicity share (4 cols): racepctblack, racePctWhite,
#     racePctAsian, racePctHisp.
#   * Per-capita income broken down by race (6 cols): whitePerCap,
#     blackPerCap, indianPerCap, AsianPerCap, OtherPerCap, HispPerCap.
#   * Aggregate income / public-assistance indicators (4 cols):
#     medIncome, perCapInc, pctWInvInc, pctWPubAsst.
#
# The first two buckets are the obvious race/ethnicity columns. The
# third bucket is included because outline §5.3 calls out "race,
# ethnicity, **income**" together and the predictive-policing
# literature documents income-related features as proxies for the
# protected attributes; dropping them avoids leaking demographic
# information back in via correlated features. This is a defensible
# but not unique taxonomy — the Phase E ablation row should report
# both the SENSITIVE_COLUMNS contents AND the actual per-run
# ``dropped_sensitive_columns`` from ``partition_stats.json`` so the
# bias-mitigation claim is reproducible. Missing columns are tolerated
# silently so the loader keeps working if UCI revises the schema.
SENSITIVE_COLUMNS: tuple[str, ...] = (
    # Race/ethnicity share
    "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp",
    # Per-capita income by race (proxy for race/ethnicity)
    "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap",
    "OtherPerCap", "HispPerCap",
    # Aggregate income / public-assistance indicators
    "medIncome", "perCapInc", "pctWInvInc", "pctWPubAsst",
)

# Crime-rate columns required for label engineering. Engineering will
# error early if any are missing — drop_missing_targets() runs before
# the label step, so an absent crime-rate column means the schema has
# drifted, not that the data is sparse.
CRIME_RATE_COLUMNS: tuple[str, ...] = (
    "murdPerPop", "rapesPerPop", "robbbPerPop", "assaultPerPop",
    "burglPerPop", "larcPerPop", "autoTheftPerPop", "arsonsPerPop",
)


# ───────────────────────────────────────────────────────────────────────
#   Phase 1 — Acquisition
# ───────────────────────────────────────────────────────────────────────

def download_communities_crime(target_dir: os.PathLike | str = DEFAULT_TARGET_DIR,
                                data_url: str = DEFAULT_DATA_URL,
                                names_url: str = DEFAULT_NAMES_URL,
                                force: bool = False) -> tuple[Path, Path]:
    """Fetch the raw data + .names file into ``target_dir``.

    Idempotent: skips files that already exist unless ``force=True``.
    Returns the two paths ``(data_path, names_path)``.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    data_path = target_dir / RAW_DATA_FILENAME
    names_path = target_dir / RAW_NAMES_FILENAME

    for url, dest in ((data_url, data_path), (names_url, names_path)):
        if dest.exists() and not force:
            continue
        print(f"Downloading {url} → {dest}")
        urllib.request.urlretrieve(url, dest)

    return data_path, names_path


# ───────────────────────────────────────────────────────────────────────
#   Phase 1 — Header parsing
# ───────────────────────────────────────────────────────────────────────

_ATTRIBUTE_RE = re.compile(r"^@attribute\s+(\S+)", re.IGNORECASE)


def parse_names_file(path: os.PathLike | str) -> list[str]:
    """Parse the UCI .names file and return column names in order.

    The file is ARFF-style; every column is declared by an
    ``@attribute <name> <type>`` line. Lines that don't match are
    ignored.
    """
    names: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = _ATTRIBUTE_RE.match(line.strip())
            if match:
                names.append(match.group(1))
    if not names:
        raise ValueError(f"No @attribute lines found in {path}")
    return names


# ───────────────────────────────────────────────────────────────────────
#   Phase 1 — Load
# ───────────────────────────────────────────────────────────────────────

def load_raw(csv_path: os.PathLike | str,
             names: Iterable[str]) -> pd.DataFrame:
    """Load the raw UCI CSV with ``na_values='?'`` and the supplied headers."""
    return pd.read_csv(csv_path, header=None, names=list(names), na_values="?")


# ───────────────────────────────────────────────────────────────────────
#   Phase 2 — Schema audit
# ───────────────────────────────────────────────────────────────────────

def audit(df: pd.DataFrame) -> dict:
    """Schema audit for outline §6.2.

    Returned dict is JSON-serializable so it can be dropped into
    ``partition_stats.json`` under an ``audit`` key.
    """
    state_col = df["state"] if "state" in df.columns else None
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3),
        "missing_per_col": {c: int(df[c].isna().sum())
                            for c in df.columns if df[c].isna().any()},
        "state_value_counts": (state_col.value_counts(dropna=False).to_dict()
                               if state_col is not None else {}),
    }


# ───────────────────────────────────────────────────────────────────────
#   Phase 3 — Cleaning
# ───────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame,
          drop_sensitive: bool = True,
          drop_missing_targets: bool = True) -> pd.DataFrame:
    """Phase-3 cleaning per outline §6.3.

    1. Drop non-predictive identifier columns (``ID_COLUMNS``).
    2. Optionally drop sensitive demographic columns.
    3. Optionally drop rows missing any crime-target column.
    """
    df = df.copy()

    for col in ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    if drop_sensitive:
        present = [c for c in SENSITIVE_COLUMNS if c in df.columns]
        if present:
            df = df.drop(columns=present)

    if drop_missing_targets:
        target_cols = [c for c in CRIME_RATE_COLUMNS if c in df.columns]
        if target_cols:
            df = df.dropna(subset=target_cols).reset_index(drop=True)

    return df


# ───────────────────────────────────────────────────────────────────────
#   Phase 1.5 — Synthetic stub (offline tests)
# ───────────────────────────────────────────────────────────────────────

# Minimal column set that the downstream pipeline expects. Includes
# ``state`` (for partitioning), the crime-rate columns (for label
# engineering), and a handful of feature columns so scaling is non-trivial.
_STUB_COLUMNS: tuple[str, ...] = (
    "communityname", "state", "countyCode", "communityCode", "fold",
    "population", "householdsize", "racepctblack", "racePctWhite",
    "agePct12t21", "agePct65up", "medIncome", "pctWInvInc",
    "PctUnemployed", "PctEmploy", "PctIlleg", "PctLargHouseFam",
    "HousVacant", "PctHousOccup", "MedRent", "NumStreet",
    *CRIME_RATE_COLUMNS,
)

# FIPS state codes that match the partitioner's region buckets; the
# stub deliberately spans all 5 N=5 regions so geographic partitioning
# is testable end-to-end on the stub alone.
_STUB_STATE_CODES = (6, 12, 17, 36, 48, 25, 39, 13, 51, 22)  # CA, FL, IL, NY, TX, MA, OH, GA, VA, LA


def make_synthetic_stub(out_path: os.PathLike | str,
                         n_rows: int = 50,
                         seed: int = 42) -> Path:
    """Write a deterministic synthetic CSV that matches the UCI schema.

    Used by Phase A unit + integration tests. Not a realistic dataset —
    just enough columns and structure for the loader → label engineer →
    partitioner pipeline to exercise every code path offline.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    n_features = len(_STUB_COLUMNS) - 5 - len(CRIME_RATE_COLUMNS)
    feature_block = rng.uniform(0.0, 1.0, size=(n_rows, n_features))
    crime_block = rng.uniform(0.0, 5000.0, size=(n_rows, len(CRIME_RATE_COLUMNS)))
    states = rng.choice(_STUB_STATE_CODES, size=n_rows)

    df = pd.DataFrame({
        "communityname": [f"stub_city_{i:04d}" for i in range(n_rows)],
        "state": states,
        "countyCode": rng.integers(1, 999, size=n_rows),
        "communityCode": rng.integers(1, 99999, size=n_rows),
        "fold": rng.integers(1, 11, size=n_rows),
    })
    feat_names = [c for c in _STUB_COLUMNS
                  if c not in df.columns and c not in CRIME_RATE_COLUMNS]
    for i, name in enumerate(feat_names):
        df[name] = feature_block[:, i]
    for i, name in enumerate(CRIME_RATE_COLUMNS):
        df[name] = crime_block[:, i]

    df.to_csv(out_path, header=False, index=False)
    return out_path


def stub_names_file(out_path: os.PathLike | str) -> Path:
    """Write a .names file matching ``make_synthetic_stub``'s column order."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"@attribute {c} numeric" if c != "communityname"
             else "@attribute communityname string"
             for c in _STUB_COLUMNS]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
