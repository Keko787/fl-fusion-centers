"""Phase D.1 — FusionFedProx strategy tests.

Pins:
* ``configure_fit`` broadcasts ``mu`` in every client's FitIns config.
* All Phase C ``FusionFedAvg`` behavior is inherited
  (aggregate_evaluate, aggregate_fit stash, plateau detection, NaN filter).
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from unittest.mock import MagicMock

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.FusionFedAvgConfig import (
    FusionFedAvg,
)
from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.FusionFedProxConfig import (
    FusionFedProx,
)


def _mock_client_manager(num_clients: int = 2):
    mgr = MagicMock()
    mgr.num_available.return_value = num_clients
    mgr.sample.return_value = [MagicMock() for _ in range(num_clients)]
    return mgr


def test_fedprox_inherits_from_fedavg():
    strategy = FusionFedProx(
        fedprox_mu=0.01,
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    assert isinstance(strategy, FusionFedAvg)
    assert strategy.fedprox_mu == 0.01


def test_configure_fit_broadcasts_mu():
    params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
    strategy = FusionFedProx(
        fedprox_mu=0.05,
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    instructions = strategy.configure_fit(
        server_round=1, parameters=params,
        client_manager=_mock_client_manager(num_clients=2),
    )
    assert len(instructions) == 2
    for _, fit_ins in instructions:
        assert "mu" in fit_ins.config
        assert fit_ins.config["mu"] == pytest.approx(0.05)


def test_configure_fit_mu_zero_when_constructed_with_zero():
    """FedProx with mu=0 broadcasts mu=0 — behaviorally equivalent to FedAvg
    on the client side, but still uses the FedProx strategy code path."""
    params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
    strategy = FusionFedProx(
        fedprox_mu=0.0,
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
    )
    instructions = strategy.configure_fit(
        server_round=1, parameters=params,
        client_manager=_mock_client_manager(num_clients=1),
    )
    for _, fit_ins in instructions:
        assert fit_ins.config["mu"] == 0.0


def test_fedprox_aggregate_fit_stashes_parameters():
    """Inherited from FusionFedAvg — verify the override still works
    through the subclass."""
    strategy = FusionFedProx(
        fedprox_mu=0.01,
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
    )
    assert strategy._final_parameters is None

    w1 = [np.array([[1.0, 2.0]], dtype=np.float32)]
    w2 = [np.array([[3.0, 4.0]], dtype=np.float32)]
    results = [
        (MagicMock(), FitRes(status=Status(Code.OK, "ok"),
                              parameters=ndarrays_to_parameters(w1),
                              num_examples=10, metrics={})),
        (MagicMock(), FitRes(status=Status(Code.OK, "ok"),
                              parameters=ndarrays_to_parameters(w2),
                              num_examples=10, metrics={})),
    ]
    strategy.aggregate_fit(1, results, [])
    assert strategy._final_parameters is not None
