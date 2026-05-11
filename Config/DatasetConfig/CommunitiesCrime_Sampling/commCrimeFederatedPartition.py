"""Federated partitioning for Communities and Crime (Phase A.3 / outline §6.7).

Three strategies:

* ``geographic`` — primary. State FIPS codes are mapped to one of N
  regional buckets (Northeast / Southeast / Midwest / Southwest / West
  for N=5; collapsed for N=3; sub-divided for N=10).
* ``iid`` — random shuffle into N equal partitions. Used as the
  baseline experiment (the "easy case" sanity check).
* ``dirichlet`` — per-class Dirichlet allocation with concentration
  ``alpha``. Used for ablations beyond the headline experiments.

The global test split is held out **before** any per-client split so
every model configuration evaluates on the same data (outline §6.6).
On first call the global test is frozen to disk; subsequent calls with
the same run directory re-load it.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# FIPS state code → 2-letter postal abbreviation. Only US states +
# DC; UCI Communities and Crime doesn't include territories.
FIPS_TO_POSTAL: dict[int, str] = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL",
    18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
    25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND",
    39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}

# N=5 buckets (outline §6.7 default).
_BUCKETS_N5: dict[int, tuple[str, ...]] = {
    0: ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),  # Northeast
    1: ("DE", "MD", "DC", "VA", "WV", "NC", "SC", "GA", "FL",
        "KY", "TN", "AL", "MS", "AR", "LA"),  # Southeast
    2: ("OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO",
        "ND", "SD", "NE", "KS"),  # Midwest
    3: ("TX", "OK", "NM", "AZ"),  # Southwest
    4: ("CO", "WY", "MT", "ID", "UT", "NV", "CA", "OR", "WA", "AK", "HI"),  # West
}

# N=3 buckets — collapse the N=5 partitioning.
_BUCKETS_N3: dict[int, tuple[str, ...]] = {
    0: _BUCKETS_N5[0] + _BUCKETS_N5[1],  # East
    1: _BUCKETS_N5[2] + _BUCKETS_N5[3],  # Central
    2: _BUCKETS_N5[4],                    # West
}

# N=10 buckets — Census Bureau 9 divisions with the South Atlantic split.
_BUCKETS_N10: dict[int, tuple[str, ...]] = {
    0: ("CT", "ME", "MA", "NH", "RI", "VT"),                       # New England
    1: ("NJ", "NY", "PA"),                                          # Mid-Atlantic
    2: ("DE", "MD", "DC", "VA", "WV"),                              # South Atlantic E
    3: ("NC", "SC", "GA", "FL"),                                    # South Atlantic W
    4: ("KY", "TN", "AL", "MS"),                                    # E South Central
    5: ("OH", "IN", "IL", "MI", "WI"),                              # E North Central
    6: ("MN", "IA", "MO", "ND", "SD", "NE", "KS"),                  # W North Central
    7: ("AR", "LA", "OK", "TX"),                                    # W South Central
    8: ("NM", "AZ", "CO", "WY", "MT", "ID", "UT", "NV"),            # Mountain
    9: ("CA", "OR", "WA", "AK", "HI"),                              # Pacific
}

# N=1 — the centralized-baseline degenerate case. Every state maps to
# bucket 0; the "federation" is a single client holding the union of
# all training partitions. Plan B uses this for the centralized
# baseline experiment.
_BUCKETS_N1: dict[int, tuple[str, ...]] = {
    0: sum((postals for postals in _BUCKETS_N5.values()), ()),
}

_BUCKET_TABLES: dict[int, dict[int, tuple[str, ...]]] = {
    1: _BUCKETS_N1, 3: _BUCKETS_N3, 5: _BUCKETS_N5, 10: _BUCKETS_N10,
}


def _postal_to_bucket(num_clients: int) -> dict[str, int]:
    if num_clients not in _BUCKET_TABLES:
        raise ValueError(
            f"num_clients={num_clients} not supported; "
            f"built-in buckets exist for {sorted(_BUCKET_TABLES)}"
        )
    out: dict[str, int] = {}
    for bucket_id, postals in _BUCKET_TABLES[num_clients].items():
        for p in postals:
            out[p] = bucket_id
    return out


def _stratified_global_test(df: pd.DataFrame, frac: float,
                             seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a stratified global test split on ``threat_class``."""
    rng = np.random.default_rng(seed)
    test_idx: list[int] = []
    for _, group in df.groupby("threat_class"):
        n_test = max(1, int(round(len(group) * frac)))
        chosen = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        test_idx.extend(chosen.tolist())
    test_mask = df.index.isin(test_idx)
    return df.loc[~test_mask].reset_index(drop=True), df.loc[test_mask].reset_index(drop=True)


def _train_val_split(client_df: pd.DataFrame, val_frac: float,
                      seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(client_df)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return (client_df.iloc[train_idx].reset_index(drop=True),
            client_df.iloc[val_idx].reset_index(drop=True))


def _split_geographic(remaining: pd.DataFrame, num_clients: int,
                       state_col: str = "state") -> tuple[list[pd.DataFrame], int]:
    """Bucket rows by state into ``num_clients`` regional partitions.

    Returns ``(client_partitions, unmapped_count)``. ``unmapped_count``
    captures rows whose state was NaN or whose FIPS/postal code is not
    in the bucket table; these rows are distributed round-robin so they
    are never silently dropped but do dilute the non-IID property of
    the geographic split.
    """
    postal_to_bucket = _postal_to_bucket(num_clients)
    state_series = remaining[state_col]

    if pd.api.types.is_numeric_dtype(state_series):
        # A single NaN coerces the whole column to float64, which would
        # then miss every integer FIPS key on lookup. Cast non-null
        # entries to int explicitly and leave NaN entries as None so
        # the unknown-state fallback below catches them.
        valid = state_series.notna()
        postal = pd.Series([None] * len(state_series),
                           index=state_series.index, dtype=object)
        postal.loc[valid] = state_series.loc[valid].astype(int).map(FIPS_TO_POSTAL)
    else:
        postal = state_series.astype(str).str.upper()

    buckets = postal.map(postal_to_bucket)
    unmapped = int(buckets.isna().sum())
    if unmapped:
        print(f"⚠️  {unmapped} rows with unmapped/missing state codes; "
              f"distributing round-robin across {num_clients} clients.")
        buckets = buckets.fillna(
            pd.Series(np.arange(len(buckets)) % num_clients, index=buckets.index)
        )
    client_partitions = [remaining.loc[buckets == i].reset_index(drop=True)
                          for i in range(num_clients)]
    return client_partitions, unmapped


def _split_iid(remaining: pd.DataFrame, num_clients: int,
                seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(remaining))
    chunks = np.array_split(perm, num_clients)
    return [remaining.iloc[chunk].reset_index(drop=True) for chunk in chunks]


def _split_dirichlet(remaining: pd.DataFrame, num_clients: int,
                      alpha: float, seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    client_indices: dict[int, list[int]] = defaultdict(list)
    for _, group in remaining.groupby("threat_class"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    return [remaining.loc[client_indices[i]].reset_index(drop=True)
            for i in range(num_clients)]


def _client_stats(client_df: pd.DataFrame,
                   feature_cols: Iterable[str]) -> dict:
    feature_cols = list(feature_cols)
    return {
        "n_rows": int(len(client_df)),
        "class_distribution": (client_df["threat_class"]
                                 .value_counts().sort_index().to_dict()
                               if "threat_class" in client_df.columns else {}),
        "escalation_mean": (float(client_df["escalation_score"].mean())
                            if "escalation_score" in client_df.columns
                            and len(client_df) else None),
        "feature_mean": {c: float(client_df[c].mean())
                         for c in feature_cols if c in client_df.columns
                         and len(client_df)},
        "feature_std": {c: float(client_df[c].std())
                        for c in feature_cols if c in client_df.columns
                        and len(client_df)},
    }


_CONFLICT_KEYS: tuple[str, ...] = (
    "strategy", "num_clients", "seed", "global_test_size", "val_frac",
    "dirichlet_alpha",
)


def _check_partition_conflict(existing: dict, new: dict) -> None:
    """Raise ``ValueError`` if any partition-shape input differs from an existing run.

    Re-running ``partition()`` against the same ``run_dir`` is fine when
    every input that determines the partition shape is identical
    (byte-for-byte idempotent), but a silent overwrite when something
    differs would leave the run dir in a half-consistent state — frozen
    global test + new client partitions + new stats. Raising here keeps
    the contract honest.
    """
    mismatches = []
    for key in _CONFLICT_KEYS:
        if existing.get(key) != new.get(key):
            mismatches.append(f"  {key}: existing={existing.get(key)!r}, new={new.get(key)!r}")
    if mismatches:
        raise ValueError(
            "partition() called with inputs that conflict with the existing "
            "partition_stats.json:\n" + "\n".join(mismatches) +
            "\nUse a fresh --run_dir or delete the existing partition artifacts."
        )


def partition(labeled_df: pd.DataFrame,
              strategy: str,
              num_clients: int,
              seed: int,
              run_dir: os.PathLike | str,
              dirichlet_alpha: float | None = None,
              global_test_size: float = 0.15,
              val_frac: float = 0.20,
              state_col: str = "state",
              feature_cols_for_stats: Iterable[str] | None = None,
              audit_info: dict | None = None,
              dropped_sensitive_columns: Iterable[str] | None = None) -> dict:
    """Run the full Phase 6 partitioning per outline §6.6 and §6.7.

    Returns a dict ``{client_id: {"train": df, "val": df}}`` plus a
    ``"global_test"`` key. All partitions and a ``partition_stats.json``
    summary are persisted under ``run_dir/partitions/``.

    Re-run semantics: calling ``partition()`` twice against the same
    ``run_dir`` is idempotent when every partition-shape input matches
    the existing ``partition_stats.json`` (strategy, num_clients, seed,
    global_test_size, val_frac, dirichlet_alpha). Any mismatch raises
    ``ValueError`` rather than silently overwriting the existing
    artifacts. The frozen global test split also re-loads on idempotent
    re-runs.
    """
    run_dir = Path(run_dir)
    part_dir = run_dir / "partitions"
    part_dir.mkdir(parents=True, exist_ok=True)

    # Re-run conflict check (see _check_partition_conflict).
    stats_path = run_dir / "partition_stats.json"
    new_shape_inputs = {
        "strategy": strategy, "num_clients": num_clients, "seed": seed,
        "global_test_size": global_test_size, "val_frac": val_frac,
        "dirichlet_alpha": dirichlet_alpha,
    }
    if stats_path.exists():
        existing = json.loads(stats_path.read_text())
        _check_partition_conflict(existing, new_shape_inputs)

    global_test_path = part_dir / "global_test.pkl"
    if global_test_path.exists():
        global_test = pickle.loads(global_test_path.read_bytes())
        # Re-derive the "remaining" set as the complement on row signature;
        # since the global test was held out deterministically, we can
        # recompute it from scratch with the same seed.
        remaining, _ = _stratified_global_test(labeled_df, global_test_size, seed)
    else:
        remaining, global_test = _stratified_global_test(
            labeled_df, global_test_size, seed,
        )
        global_test_path.write_bytes(pickle.dumps(global_test))

    unmapped_state_rows = 0
    if strategy == "geographic":
        client_partitions, unmapped_state_rows = _split_geographic(
            remaining, num_clients, state_col,
        )
    elif strategy == "iid":
        client_partitions = _split_iid(remaining, num_clients, seed)
    elif strategy == "dirichlet":
        if dirichlet_alpha is None:
            raise ValueError("strategy='dirichlet' requires dirichlet_alpha")
        client_partitions = _split_dirichlet(
            remaining, num_clients, dirichlet_alpha, seed,
        )
    else:
        raise ValueError(f"Unknown partition strategy: {strategy!r}")

    if feature_cols_for_stats is None:
        # Sensible default for the audit: everything that isn't a label
        # or state column. Stats writer ignores missing columns.
        feature_cols_for_stats = [c for c in remaining.columns
                                   if c not in {"threat_class",
                                                "escalation_score",
                                                state_col}]

    stats: dict = {
        "strategy": strategy,
        "num_clients": num_clients,
        "seed": seed,
        "global_test_size": global_test_size,
        "val_frac": val_frac,
        "dirichlet_alpha": dirichlet_alpha,
        "unmapped_state_rows": unmapped_state_rows,
        "dropped_sensitive_columns": list(dropped_sensitive_columns or []),
        "audit": audit_info or {},
        "global_test_stats": _client_stats(global_test, feature_cols_for_stats),
        "clients": {},
    }

    result: dict = {"global_test": global_test}
    for i, client_df in enumerate(client_partitions):
        train_df, val_df = _train_val_split(client_df, val_frac, seed + i)
        client_path = part_dir / f"client_{i}.pkl"
        client_path.write_bytes(pickle.dumps({"train": train_df, "val": val_df}))
        result[i] = {"train": train_df, "val": val_df}
        stats["clients"][str(i)] = {
            "train": _client_stats(train_df, feature_cols_for_stats),
            "val": _client_stats(val_df, feature_cols_for_stats),
        }

    (run_dir / "partition_stats.json").write_text(
        json.dumps(stats, indent=2, default=str),
    )

    return result


def load_partition(run_dir: os.PathLike | str,
                    client_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-load (train, val, global_test) for a client from an existing run dir."""
    run_dir = Path(run_dir)
    part_dir = run_dir / "partitions"
    payload = pickle.loads((part_dir / f"client_{client_id}.pkl").read_bytes())
    global_test = pickle.loads((part_dir / "global_test.pkl").read_bytes())
    return payload["train"], payload["val"], global_test
