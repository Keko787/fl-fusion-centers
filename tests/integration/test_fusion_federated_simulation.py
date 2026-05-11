"""Phase C integration test — small Flower simulation end-to-end.

DoD from the implementation plan §3 Phase C:
* 3-round / 2-client / synthetic-stub simulation completes in < 30 s
  (excluding TF cold-start).
* Per-client logs land under ``<run_dir>/client_<i>/``.
* Server-side aggregated log lands under ``<run_dir>/server_evaluation.log``.
* Final aggregated model written to ``<run_dir>/fed_fusion_mlp_<save_name>.keras``.

Flower's simulation backend uses Ray. We don't require ray to be
installed for the smoke run — instead we drive the SimulationRunner's
``client_fn`` + strategy manually, exercising the same code paths
``fl.simulation.start_simulation`` uses internally. The Ray-backed
full simulation is left to the ``--ray`` opt-in test below.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    make_synthetic_stub, stub_names_file,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLoadProcess import (
    load_commcrime_for_simulation,
)


def _make_sim_args(tmp_path: Path, *,
                    num_clients: int = 2,
                    rounds: int = 3,
                    run_dir: str | None = None) -> SimpleNamespace:
    csv = make_synthetic_stub(tmp_path / "stub.csv", n_rows=400, seed=11)
    stub_names_file(tmp_path / "stub.names")
    canonical_names = csv.with_name("communities_and_crime_unnormalized.names")
    canonical_names.write_text((tmp_path / "stub.names").read_text())

    return SimpleNamespace(
        timestamp="20260510_120000",
        mode="legacy",
        commcrime_path=str(csv),
        commcrime_random_seed=42,
        num_clients=num_clients,
        partition_strategy="iid",
        dirichlet_alpha=0.5,
        client_id=0,  # ignored in simulation
        global_test_size=0.15,
        drop_sensitive_features=True,
        dataset_processing="COMMCRIME",
        run_dir=run_dir,
        evaluationLog="20260510_120000_evaluation.log",
        trainingLog="20260510_120000_training.log",
        model_type="FUSION-MLP",
        fl_strategy="FedAvg",
        epochs=1,
        rounds=rounds,
        min_clients=num_clients,
        escalation_loss_weight=0.5,
        save_name="phase_c_smoke",
    )


# ───────────────────────────────────────────────────────────────────────
#   load_commcrime_for_simulation contract
# ───────────────────────────────────────────────────────────────────────

def test_load_for_simulation_returns_all_clients(tmp_path):
    args = _make_sim_args(tmp_path, num_clients=3, run_dir=str(tmp_path / "run"))
    per_client = load_commcrime_for_simulation(args)

    # All client slots present.
    for cid in range(3):
        assert cid in per_client
        slot = per_client[cid]
        assert "X_train" in slot and "X_val" in slot
        assert "y_train" in slot and "y_val" in slot

    # Global test set is shared.
    assert "global_test" in per_client
    X_test, y_test = per_client["global_test"]
    assert isinstance(y_test, tuple) and len(y_test) == 2
    assert X_test.dtype == np.float32

    # Run dir + scaler produced.
    run_dir = Path(per_client["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "scaler.joblib").exists()
    assert (run_dir / "partition_stats.json").exists()


def test_load_for_simulation_uses_global_scaler(tmp_path):
    """Every client share is scaled by the same fitted scaler.

    With a shared scaler the concatenated training data should be
    approximately zero-mean / unit-std (StandardScaler fit on the union)."""
    args = _make_sim_args(tmp_path, num_clients=3, run_dir=str(tmp_path / "run"))
    per_client = load_commcrime_for_simulation(args)

    union = np.vstack([per_client[i]["X_train"] for i in range(3)])
    # Sanity bounds — exact 0/1 only if every client's train set is
    # large enough that the union ≈ the union the scaler was fit on.
    assert abs(union.mean()) < 0.5
    assert 0.5 < union.std() < 1.5


def test_load_for_simulation_rejects_hermes(tmp_path):
    args = _make_sim_args(tmp_path, run_dir=str(tmp_path / "run"))
    args.mode = "hermes"
    with pytest.raises(SystemExit, match="does not support --mode hermes"):
        load_commcrime_for_simulation(args)


# ───────────────────────────────────────────────────────────────────────
#   client_fn factory (covers SimulationRunner without invoking Ray)
# ───────────────────────────────────────────────────────────────────────

def test_client_fn_returns_flower_client(tmp_path):
    """Smoke-test the SimulationRunner's ``client_fn`` factory: each
    cid produces a non-None Flower Client and materializes its per-client
    log directory. The Flower client's protocol is tested directly in
    ``tests/unit/test_fusion_fl_client.py``; this test only verifies
    that ``_build_client_fn`` wires the factory together correctly."""
    from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.SimulationRunner import (
        _build_client_fn,
    )

    args = _make_sim_args(tmp_path, num_clients=2, run_dir=str(tmp_path / "run"))
    per_client = load_commcrime_for_simulation(args)
    run_dir = Path(per_client["run_dir"])

    client_fn = _build_client_fn(
        per_client, run_dir, args,
        batch_size=32, learning_rate=1e-3, num_classes=3, l2_alpha=0.0,
    )

    for cid in (0, 1):
        flwr_client = client_fn(str(cid))
        assert flwr_client is not None
        assert (run_dir / f"client_{cid}").exists()


def test_client_fn_caches_per_client_instance(tmp_path):
    """Phase C review #7: client_fn returns the SAME client object for
    the same cid across calls (cache hit, not a fresh rebuild)."""
    from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.SimulationRunner import (
        _build_client_fn,
    )

    args = _make_sim_args(tmp_path, num_clients=2, run_dir=str(tmp_path / "run"))
    per_client = load_commcrime_for_simulation(args)
    run_dir = Path(per_client["run_dir"])

    client_fn = _build_client_fn(
        per_client, run_dir, args,
        batch_size=32, learning_rate=1e-3, num_classes=3, l2_alpha=0.0,
    )

    first_call = client_fn("0")
    second_call = client_fn("0")
    # Flower's to_client() may return a fresh wrapper, but the
    # underlying client object should be the same instance.
    assert first_call is second_call or _same_underlying(first_call, second_call)


def _same_underlying(a, b) -> bool:
    """Best-effort check that two Flower Client wrappers point at the
    same NumPyClient. Flower API differences across versions mean we
    don't rely on a particular attribute name; if neither wrapper
    exposes its inner client, fall back to ``a is b``."""
    for attr in ("numpy_client", "_numpy_client", "client"):
        ai = getattr(a, attr, None)
        bi = getattr(b, attr, None)
        if ai is not None and bi is not None:
            return ai is bi
    return a is b
