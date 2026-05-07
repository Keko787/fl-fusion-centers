"""Experiment-3 trial-grid driver — selects and runs the right arm.

Plugs into :class:`~experiments.runner.TrialRunner`'s
``run_trial(cell)`` slot. The cell's ``arm`` field selects A1 / A2 /
A3 / A4; the cell's ``params`` carries the sweep coordinates (N, β,
deadline_het, rrf).

Returns a metric dict the runner writes as one CSV row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from experiments.calibration import Exp3Calibration, load_calibration
from experiments.runner import Cell

from hermes.scheduler.policies import (
    ArrivalOrderPolicy,
    EdfFeasibilityPolicy,
)
from hermes.scheduler.policies.edf_feasibility import FeasibilityModel
from hermes.scheduler.selector import TargetSelectorRL

from .arm_a1 import A1Config, run_a1_trial
from .arm_mule import MuleArmConfig, run_mule_trial
from .metrics import Exp3MetricSummary
from .sim_env import Exp3SimConfig

log = logging.getLogger("experiments.exp3.driver")


ARMS = ("A1", "A2", "A3", "A4")


@dataclass
class Exp3Driver:
    """Owns the per-trial dispatch + the (optional) shared A4 selector.

    ``upload_bytes_per_contact`` and ``nominal_upload_bps`` set the
    sim's upload model. The defaults (10 MB at 1 Mbps) put genuine
    pressure on A3's feasibility filter; legacy "calibration-free"
    runs used 0 bytes and had a toothless filter, so the runner_main
    CLI and any caller wanting paper-grade results should pass
    realistic values.
    """

    calibration: Optional[Exp3Calibration] = None
    selector_a4: Optional[TargetSelectorRL] = None
    upload_bytes_per_contact: float = 1.0e7
    nominal_upload_bps: float = 1.0e6
    # Default jittery-mode parameters at the driver level. Per-cell
    # overrides via ``cell.params['jittery']`` (bool) flip these to
    # the Exp.\ 1 jittery values (2% loss, 30% latency jitter) for
    # that cell. See ``runner_main.py``'s ``--jittery`` axis.
    packet_loss_pct: float = 0.0
    latency_jitter_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.calibration is None:
            try:
                self.calibration = load_calibration().exp3
            except Exception as e:  # pragma: no cover - calibration optional
                log.warning("calibration load failed (energy ⟶ None): %s", e)
                self.calibration = None
        if self.selector_a4 is None:
            # An untrained selector still ranks deterministically (the
            # initial random init biases the order — tests can pass an
            # explicitly trained one for paper-grade runs).
            self.selector_a4 = TargetSelectorRL(epsilon=0.0)

    # ------------------------------------------------------------------ #
    # Trial dispatch
    # ------------------------------------------------------------------ #

    def run_trial(self, cell: Cell) -> Mapping[str, Any]:
        params = cell.params
        arm = cell.arm
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

        if arm == "A1":
            summary = self._run_a1(cell, params)
        else:
            summary = self._run_mule(cell, params, arm)

        row: Dict[str, Any] = dict(summary.to_row())
        row["n_devices"] = int(params.get("N", params.get("n_devices", 10)))
        row["beta"] = float(params.get("beta", 1.0))
        row["deadline_het"] = bool(params.get("deadline_het", False))
        row["rf_range_m"] = float(params.get("rrf", params.get("rf_range_m", 60.0)))
        row["jittery"] = bool(params.get("jittery", False))
        return row

    # ------------------------------------------------------------------ #
    # Per-arm runners
    # ------------------------------------------------------------------ #

    def _run_a1(self, cell: Cell, params: Mapping[str, Any]) -> Exp3MetricSummary:
        n = int(params.get("N", params.get("n_devices", 10)))
        beta = float(params.get("beta", 1.0))
        # The A1 mean round time scales with β: lower β tightens the
        # deadline but doesn't change wall-clock; the deadline scales
        # via ``round_deadline_s = base * beta``.
        cfg = A1Config(
            n_clients=n,
            n_rounds=int(params.get("n_rounds_a1", 20)),
            client_fraction=float(params.get("client_fraction", 1.0)),
            round_deadline_s=float(params.get("round_deadline_s", 60.0)) * beta,
            seed=cell.seed,
        )
        return run_a1_trial(cfg)

    def _run_mule(
        self, cell: Cell, params: Mapping[str, Any], arm: str,
    ) -> Exp3MetricSummary:
        # Jittery cells flip the Exp.\ 1-parity noise on (2% packet
        # loss + 30% latency jitter). Clean cells leave it at the
        # driver default (typically 0).
        jittery = bool(params.get("jittery", False))
        if jittery:
            packet_loss = max(self.packet_loss_pct, 2.0)
            latency_jitter = max(self.latency_jitter_pct, 30.0)
        else:
            packet_loss = self.packet_loss_pct
            latency_jitter = self.latency_jitter_pct

        sim_cfg = Exp3SimConfig(
            n_devices=int(params.get("N", params.get("n_devices", 10))),
            beta=float(params.get("beta", 1.0)),
            deadline_heterogeneity=bool(params.get("deadline_het", False)),
            rf_range_m=float(params.get("rrf", params.get("rf_range_m", 60.0))),
            mission_budget_s=float(params.get("mission_budget_s", 600.0)),
            upload_bytes_per_contact=float(
                params.get("upload_bytes_per_contact",
                           self.upload_bytes_per_contact)
            ),
            nominal_upload_bps=float(
                params.get("nominal_upload_bps", self.nominal_upload_bps)
            ),
            packet_loss_pct=packet_loss,
            latency_jitter_pct=latency_jitter,
            seed=cell.seed,
        )
        policy = self._policy_for(arm, sim_cfg)
        mule_cfg = MuleArmConfig(arm_name=arm, sim=sim_cfg)
        return run_mule_trial(cfg=mule_cfg, policy=policy, cal=self.calibration)

    def _policy_for(self, arm: str, sim_cfg: Exp3SimConfig):
        if arm == "A2":
            return ArrivalOrderPolicy()
        if arm == "A3":
            model = FeasibilityModel(
                cruise_speed_m_s=sim_cfg.cruise_speed_m_s,
                session_time_s=sim_cfg.session_time_s,
                base_station_positions=sim_cfg.base_station_positions,
                upload_bytes_per_contact=sim_cfg.upload_bytes_per_contact,
                nominal_upload_bps=sim_cfg.nominal_upload_bps,
                default_mission_budget_s=sim_cfg.mission_budget_s * sim_cfg.beta,
            )
            return EdfFeasibilityPolicy(model=model)
        if arm == "A4":
            assert self.selector_a4 is not None
            return self.selector_a4
        raise ValueError(f"unknown mule arm {arm!r}")
