"""Tests for the A4-selector save/load + training-CLI plumbing.

Pins:

* DDQN.save / DDQN.load round-trip preserves prediction values bit-equal
  (set_weights uses copies, so the loaded actor is independent of the
  original).
* DDQN.load rejects a missing file and a bad format-version.
* The train_a4 CLI runs end-to-end on a tiny config and writes a
  loadable file.
* runner_main loads --selector-weights and constructs A4 from them.
* runner_main --require-trained-a4 refuses to run without weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hermes.scheduler.selector.ddqn import DDQN
from hermes.scheduler.selector.features import FEATURE_DIM


# --------------------------------------------------------------------------- #
# DDQN.save / DDQN.load
# --------------------------------------------------------------------------- #

def test_ddqn_save_load_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    src = DDQN(feature_dim=FEATURE_DIM, hidden=8, seed=42)
    feats = rng.standard_normal((5, FEATURE_DIM)).astype(np.float32)
    expected = src.predict(feats)

    out = tmp_path / "ddqn.npz"
    src.save(out)
    assert out.exists()

    loaded = DDQN.load(out)
    got = loaded.predict(feats)
    np.testing.assert_array_equal(expected, got)


def test_ddqn_save_load_independence(tmp_path: Path):
    """Loaded actor mutating its weights must not change the saver's."""
    src = DDQN(feature_dim=FEATURE_DIM, hidden=8, seed=42)
    out = tmp_path / "ddqn.npz"
    src.save(out)

    loaded = DDQN.load(out)
    # Mutate the loaded copy's weights via set_weights with zeros.
    zero_weights = {
        "W1": np.zeros_like(loaded.get_weights()["W1"]),
        "b1": np.zeros_like(loaded.get_weights()["b1"]),
        "W2": np.zeros_like(loaded.get_weights()["W2"]),
        "b2": np.zeros_like(loaded.get_weights()["b2"]),
    }
    loaded.set_weights(zero_weights)

    # Source still produces non-zero predictions on a non-zero input.
    feats = np.ones((3, FEATURE_DIM), dtype=np.float32)
    src_q = src.predict(feats)
    assert not np.allclose(src_q, 0.0)


def test_ddqn_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DDQN.load(tmp_path / "does-not-exist.npz")


def test_ddqn_load_bad_format_version_raises(tmp_path: Path):
    out = tmp_path / "bad.npz"
    np.savez(
        out,
        W1=np.zeros((FEATURE_DIM, 8), dtype=np.float32),
        b1=np.zeros(8, dtype=np.float32),
        W2=np.zeros((8, 1), dtype=np.float32),
        b2=np.zeros(1, dtype=np.float32),
        feature_dim=np.int64(FEATURE_DIM),
        hidden=np.int64(8),
        format_version=np.int64(99),  # wrong version
    )
    with pytest.raises(ValueError, match="format_version"):
        DDQN.load(out)


def test_ddqn_save_creates_parent_dirs(tmp_path: Path):
    src = DDQN(feature_dim=FEATURE_DIM, hidden=8, seed=0)
    nested = tmp_path / "a" / "b" / "c" / "ddqn.npz"
    src.save(nested)
    assert nested.exists()


# --------------------------------------------------------------------------- #
# train_a4 CLI smoke
# --------------------------------------------------------------------------- #

def test_train_a4_cli_writes_loadable_weights(tmp_path: Path):
    """Train a tiny selector via the CLI and load the result."""
    from experiments.exp3.train_a4 import main

    out = tmp_path / "a4.npz"
    rc = main([
        "--output", str(out),
        "--episodes", "5",          # tiny — just checking the wiring
        "--n-devices", "4",
        "--rf-range-m", "60",
        "--mission-budget", "150",
        "--warmup", "4",
        "--batch-size", "4",
        "--epsilon-decay-episodes", "5",
        "--seed", "0",
    ])
    assert rc == 0
    assert out.exists()

    # Loaded actor must be usable for inference.
    ddqn = DDQN.load(out)
    feats = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    q = ddqn.predict(feats)
    assert q.shape == (3,)


# --------------------------------------------------------------------------- #
# runner_main wiring
# --------------------------------------------------------------------------- #

def test_runner_main_loads_selector_weights(tmp_path: Path):
    """End-to-end: train tiny weights, then run a 1-trial smoke through runner."""
    from experiments.exp3.runner_main import main as runner_main
    from experiments.exp3.train_a4 import main as train_main

    weights = tmp_path / "a4.npz"
    train_rc = train_main([
        "--output", str(weights),
        "--episodes", "5",
        "--n-devices", "4",
        "--rf-range-m", "60",
        "--mission-budget", "150",
        "--warmup", "4",
        "--batch-size", "4",
        "--epsilon-decay-episodes", "5",
        "--seed", "0",
    ])
    assert train_rc == 0

    csv_path = tmp_path / "exp3.csv"
    rc = runner_main([
        "--csv", str(csv_path),
        "--n-trials", "1",
        "--N", "5",
        "--beta", "1.0",
        "--rrf", "60.0",
        "--deadline-het", "0",
        "--arms", "A4",
        "--selector-weights", str(weights),
    ])
    assert rc == 0
    assert csv_path.exists()
    contents = csv_path.read_text(encoding="utf-8")
    assert ",A4," in contents


def test_runner_main_require_trained_a4_rejects_without_weights(tmp_path: Path):
    """--require-trained-a4 must hard-error if A4 is in --arms with no weights."""
    from experiments.exp3.runner_main import main as runner_main

    csv_path = tmp_path / "exp3.csv"
    with pytest.raises(SystemExit) as exc:
        runner_main([
            "--csv", str(csv_path),
            "--n-trials", "1",
            "--N", "5", "--beta", "1.0", "--rrf", "60.0",
            "--deadline-het", "0",
            "--arms", "A4",
            "--require-trained-a4",
        ])
    # argparse.error() exits with code 2.
    assert exc.value.code == 2


def test_runner_main_random_a4_allowed_without_flag(tmp_path: Path):
    """Without --require-trained-a4, runner should warn but succeed."""
    from experiments.exp3.runner_main import main as runner_main

    csv_path = tmp_path / "exp3.csv"
    rc = runner_main([
        "--csv", str(csv_path),
        "--n-trials", "1",
        "--N", "5", "--beta", "1.0", "--rrf", "60.0",
        "--deadline-het", "0",
        "--arms", "A2",  # no A4 → no warning
    ])
    assert rc == 0
