"""Phase A.7 — run-artifact helper tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from Config.SessionConfig.runArtifacts import (
    dump_pip_freeze, init_run_dir, resolve_run_dir, seed_all,
)


def test_init_run_dir_creates_partitions_subdir(tmp_path):
    rd = init_run_dir(base=tmp_path, prefix="testrun")
    assert rd.exists()
    assert (rd / "partitions").exists()
    assert rd.name.startswith("testrun_")


def test_init_run_dir_honors_explicit_timestamp(tmp_path):
    rd = init_run_dir(base=tmp_path, prefix="testrun", timestamp="20260101_120000")
    assert rd.name == "testrun_20260101_120000"


def test_dump_pip_freeze_writes_file(tmp_path):
    out = tmp_path / "freeze.txt"
    dump_pip_freeze(out)
    assert out.exists()
    assert out.read_text().strip() != ""


def test_seed_all_sets_python_hash_and_random():
    import random
    seed_all(123)
    a = random.random()
    seed_all(123)
    b = random.random()
    assert a == b
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_seed_all_reproducible_numpy():
    import numpy as np
    seed_all(7)
    a = np.random.rand(5)
    seed_all(7)
    b = np.random.rand(5)
    assert (a == b).all()


# ───────────────────────────────────────────────────────────────────────
#   resolve_run_dir (added per Phase A review issue #1)
# ───────────────────────────────────────────────────────────────────────

def test_resolve_run_dir_with_explicit_path_reuses_it(tmp_path):
    target = tmp_path / "myrun"
    rd = resolve_run_dir(explicit_path=str(target))
    assert rd == target
    assert (rd / "partitions").exists()


def test_resolve_run_dir_creates_missing_subdir(tmp_path):
    target = tmp_path / "another"
    target.mkdir()  # exists but partitions/ does not
    rd = resolve_run_dir(explicit_path=str(target))
    assert (rd / "partitions").exists()


def test_resolve_run_dir_without_explicit_path_falls_back_to_timestamped(tmp_path):
    rd = resolve_run_dir(None, base=tmp_path, prefix="testrun",
                         timestamp="20260101_000000")
    assert rd.name == "testrun_20260101_000000"
    assert (rd / "partitions").exists()


def test_resolve_run_dir_idempotent_on_reuse(tmp_path):
    target = tmp_path / "shared"
    rd_a = resolve_run_dir(explicit_path=str(target))
    (rd_a / "marker.txt").write_text("present")
    rd_b = resolve_run_dir(explicit_path=str(target))
    assert (rd_b / "marker.txt").read_text() == "present"
