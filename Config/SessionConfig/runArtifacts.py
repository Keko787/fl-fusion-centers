"""Reproducibility & run-artifact helpers for the fusion-centers pipeline.

Phase A.7 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.
Used by the COMMCRIME data path, the FUSION-MLP trainers, and the simulation
runner. Existing IoT/GAN paths do not import this module.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def init_run_dir(base: os.PathLike | str = "results",
                 prefix: str = "commcrime_run",
                 timestamp: str | None = None) -> Path:
    """Create ``<base>/<prefix>_<timestamp>/`` with a ``partitions/`` subdir.

    If ``timestamp`` is omitted, one is generated at call time. Pass it
    explicitly when multiple processes need to share a run directory
    (e.g. host + client in real-multi-process deployment).
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{prefix}_{ts}"
    (run_dir / "partitions").mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_run_dir(explicit_path: os.PathLike | str | None = None,
                    *,
                    base: os.PathLike | str = "results",
                    prefix: str = "commcrime_run",
                    timestamp: str | None = None) -> Path:
    """Return the run dir to use, either an explicit user-supplied path or a fresh one.

    Phase C will need N client processes to share one run directory so
    they all see the same global test split. Pass the explicit path via
    ``--run_dir``; if omitted, a new timestamped dir is created via
    :func:`init_run_dir`. The ``partitions/`` subdir is always
    materialized so callers can assume it exists.
    """
    if explicit_path:
        run_dir = Path(explicit_path)
        (run_dir / "partitions").mkdir(parents=True, exist_ok=True)
        return run_dir
    return init_run_dir(base=base, prefix=prefix, timestamp=timestamp)


def dump_pip_freeze(out_path: os.PathLike | str) -> None:
    """Write ``pip freeze`` output to ``out_path``.

    Best-effort: if the subprocess fails the file still gets written, but
    with an error stub instead of the package list. Never raises so a
    bad environment cannot block a training run.
    """
    out_path = Path(out_path)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        out_path.write_text(result.stdout or "# pip freeze produced no output\n")
    except Exception as exc:  # noqa: BLE001 — reproducibility artifact, never block
        out_path.write_text(f"# pip freeze unavailable: {exc}\n")


def seed_all(seed: int) -> None:
    """Seed every stochastic source the fusion-centers pipeline touches.

    Imports of numpy / tensorflow are guarded so this can be called from
    pure-data-engineering contexts (Phase A tests) without forcing TF
    initialization. The guards are not dead code: TF is intentionally
    absent in some CI shapes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
