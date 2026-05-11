"""Log parsers for fusion-centers experiment outputs.

Phase E.4 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

The fusion-centers simulation produces three log shapes the plot
scripts consume:

  * **Server evaluation log** ``<run_dir>/server_evaluation.log`` —
    one block per round of training (``FusionFedAvg._append_log``):

        === Round R (N clients) ===
        aggregated_loss: 0.4231
        threat_macro_f1: 0.6234
        ...

  * **Per-client evaluation log** ``<run_dir>/client_<i>/evaluation.log``
    — one block per evaluation call:

        Node|N| Round: X
        Evaluation Time Elapsed: 0.42 seconds
        threat_macro_f1: 0.62
        ...

  * **Partition stats** ``<run_dir>/partition_stats.json`` — single
    JSON file with audit, per-client distribution, dropped sensitive
    columns, etc.

This module turns each into a tidy ``pandas.DataFrame`` (or dict)
that the plot scripts can consume directly. Pure stdlib + pandas;
no TF or Flower imports.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


_SERVER_ROUND_RE = re.compile(r"^=== Round (\d+) \((\d+) clients\) ===\s*$")
_KEY_VALUE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$")


def parse_server_log(path: str | Path) -> pd.DataFrame:
    """Parse a ``FusionFedAvg`` server_evaluation.log into a tidy DataFrame.

    Returned DataFrame has one row per round and columns matching the
    metrics the strategy wrote that round (round, n_clients, aggregated_loss,
    threat_macro_f1, escalation_mae, escalation_auroc, escalation_spearman,
    fairness_macro_f1_variance, fairness_accuracy_variance, round_seconds,
    parameter_update_wire_bytes, proximal_contribution,
    plateau_detected, …).
    """
    rows: List[dict] = []
    current: dict | None = None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Server log not found: {path}")

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        round_match = _SERVER_ROUND_RE.match(line)
        if round_match:
            if current is not None:
                rows.append(current)
            current = {
                "round": int(round_match.group(1)),
                "n_clients": int(round_match.group(2)),
            }
            continue
        kv_match = _KEY_VALUE_RE.match(line)
        if kv_match and current is not None:
            key, val = kv_match.group(1), kv_match.group(2).strip()
            try:
                current[key] = float(val)
            except ValueError:
                # Non-numeric (e.g., "ok"); skip.
                pass
    if current is not None:
        rows.append(current)
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["round", "n_clients"])


def parse_client_log(path: str | Path) -> pd.DataFrame:
    """Parse a per-client ``evaluation.log`` into a tidy DataFrame.

    One row per evaluation block. ``round`` and ``node`` come from the
    ``Node|N| Round: X`` header line.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Client log not found: {path}")

    rows: List[dict] = []
    current: dict | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        header = re.match(r"^Node\|(\d+)\|\s*Round:\s*(\d+)\s*$", line)
        if header:
            if current is not None:
                rows.append(current)
            current = {
                "node": int(header.group(1)),
                "round": int(header.group(2)),
            }
            continue
        kv_match = _KEY_VALUE_RE.match(line)
        if kv_match and current is not None:
            key, val = kv_match.group(1), kv_match.group(2).strip()
            try:
                current[key] = float(val)
            except ValueError:
                pass
    if current is not None:
        rows.append(current)
    return pd.DataFrame(rows).sort_values(["node", "round"]).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["node", "round"])


def parse_partition_stats(path: str | Path) -> dict:
    """Load the ``partition_stats.json`` produced by the partitioner.

    Wrapper for symmetry + future migration if the file format evolves.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"partition_stats.json not found: {path}")
    return json.loads(path.read_text())


def collect_server_logs(run_dirs: Iterable[str | Path],
                        labels: Iterable[str] | None = None) -> pd.DataFrame:
    """Read multiple server logs and stack them with a ``label`` column.

    Used by ``plot_centralized_vs_federated`` to overlay several runs.
    ``labels`` aligns 1:1 with ``run_dirs``; if omitted, uses the run-dir
    basename.
    """
    run_dirs = list(run_dirs)
    if labels is None:
        labels = [Path(d).name for d in run_dirs]
    labels = list(labels)
    if len(labels) != len(run_dirs):
        raise ValueError("labels and run_dirs must have the same length")

    frames = []
    for run_dir, label in zip(run_dirs, labels):
        log = Path(run_dir) / "server_evaluation.log"
        df = parse_server_log(log)
        df["label"] = label
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["round", "label"])
    return pd.concat(frames, ignore_index=True)
