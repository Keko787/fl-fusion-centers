"""Experiment 3 — Centralized FL vs mule heuristics vs HERMES (A1–A4 ablation).

Sub-package layout:

* :mod:`experiments.exp3.sim_env` — Experiment-3 simulator: 3 base
  stations, time-varying per-channel upload rates, fixed-speed mule,
  configurable ``N`` / ``β`` / deadline-heterogeneity / ``rrf``.
* :mod:`experiments.exp3.metrics` — Jain's index, participation
  entropy, propulsion energy (Eq. 5), ρ_contact, round close rate,
  update yield, coverage.
* :mod:`experiments.exp3.arm_a1` — Centralized-FL-with-uniform-sampling
  driver. No mule; mule-specific metrics emitted as N/A.
* :mod:`experiments.exp3.arm_mule` — Shared mule arm driver wrapping
  :class:`~hermes.scheduler.policies.ArrivalOrderPolicy`,
  :class:`~hermes.scheduler.policies.EdfFeasibilityPolicy`, and
  :class:`~hermes.scheduler.selector.TargetSelectorRL` against the
  Experiment-3 sim env.
* :mod:`experiments.exp3.driver` — Trial-grid driver that selects the
  right arm-specific runner per :class:`Cell`.
* :mod:`experiments.exp3.runner_main` — CLI entry point that walks the
  full A1×A2×A3×A4 grid via the EX-0 harness and writes a CSV.
"""

from __future__ import annotations

from .sim_env import (
    Exp3SelectorEnv,
    Exp3Sim,
    Exp3SimConfig,
    Exp3StepResult,
)
from .metrics import (
    Exp3MetricSummary,
    Exp3RoundLog,
    aggregate_round_logs,
    coverage,
    jains_fairness,
    participation_entropy,
    round_close_rate,
    update_yield,
)

__all__ = [
    "Exp3SelectorEnv",
    "Exp3Sim",
    "Exp3SimConfig",
    "Exp3StepResult",
    "Exp3MetricSummary",
    "Exp3RoundLog",
    "aggregate_round_logs",
    "coverage",
    "jains_fairness",
    "participation_entropy",
    "round_close_rate",
    "update_yield",
]
