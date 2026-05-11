"""Server-side FedProx strategy for the fusion-centers FL update.

Phase D.1 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

Inherits all of :class:`FusionFedAvg`'s evaluate-side aggregation
(weighted means with NaN filtering, fairness variance, plateau
detection, captured ``_final_parameters``). The only difference is on
the fit-side: ``configure_fit`` broadcasts the proximal coefficient
``mu`` to clients via the per-round config dict. Clients with the
:class:`FedProxFusionMLPModel` subclass pick this up and add the
``(mu/2) * ||w - w_global||²`` term to the loss inside ``train_step``.

The aggregation logic (``aggregate_fit``) is identical to FedAvg —
the proximal term is entirely client-side, so the server's view of
the parameters is unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.FusionFedAvgConfig import (
    FusionFedAvg,
)


class FusionFedProx(FusionFedAvg):
    """FedProx variant of FusionFedAvg — adds per-round ``mu`` broadcast."""

    def __init__(self, *, fedprox_mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.fedprox_mu: float = float(fedprox_mu)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Broadcast the standard FedAvg fit config plus the proximal ``mu``.

        Clients read ``config["mu"]`` and inject the proximal term into
        their training-step loss when the value is positive.
        """
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            # Augment in place — keep any keys the base configure_fit
            # already set (e.g. ``on_fit_config_fn`` output).
            fit_ins.config["mu"] = self.fedprox_mu
        return instructions
