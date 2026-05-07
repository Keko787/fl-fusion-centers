"""Shared mule-arm driver for A2 / A3 / A4.

Wraps :class:`~experiments.exp3.sim_env.Exp3Sim` and a
ranking-policy callable with the ``rank_contacts(...) -> List[ContactWaypoint]``
shape — the API both A2/A3 (in :mod:`hermes.scheduler.policies`) and
A4 (:meth:`hermes.scheduler.selector.TargetSelectorRL.rank_contacts`)
already expose.

One driver, three arms — what changes between calls is only the
``policy`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from hermes.types import ContactWaypoint, DeviceID, MissionPass

from experiments.calibration import Exp3Calibration

from .metrics import Exp3MetricSummary, Exp3RoundLog, summarise_trial
from .sim_env import Exp3Sim, Exp3SimConfig


# --------------------------------------------------------------------------- #
# Policy protocol — what every mule arm exposes
# --------------------------------------------------------------------------- #

class ContactRankingPolicy(Protocol):
    """Subset of the selector / policy surface this driver invokes."""

    def rank_contacts(  # pragma: no cover - protocol
        self,
        candidates: Sequence[ContactWaypoint],
        device_states,
        env,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted=None,
    ) -> List[ContactWaypoint]: ...


# --------------------------------------------------------------------------- #
# Per-trial driver
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MuleArmConfig:
    """Trial-level config the driver passes to :class:`Exp3Sim`.

    ``arm_name`` is informational only — it's the column the CSV
    writes; the policy itself is what actually drives behaviour.
    """

    arm_name: str
    sim: Exp3SimConfig
    n_rounds: int = 1  # one mission = one round in the contact-event view


def run_mule_trial(
    *,
    cfg: MuleArmConfig,
    policy: ContactRankingPolicy,
    cal: Optional[Exp3Calibration] = None,
) -> Exp3MetricSummary:
    """Run one mule trial under the given ranking policy.

    Pass-1 loop:
      1. ``sim.candidates()`` -> the eligible contact list this step.
      2. ``policy.rank_contacts(...)`` -> a *re-ordered* (and possibly
         shorter, for A3's feasibility skip) list.
      3. Take the head, ``sim.step(contact)``.
      4. Stop when ``sim.done`` or the policy returns an empty list.

    Pass-2: greedy nearest-first walk over remaining contacts to
    record the Pass-2 deliverability metric.
    """
    sim = Exp3Sim(cfg.sim)
    sim.reset()

    rounds: List[Exp3RoundLog] = []
    n_devices = cfg.sim.n_devices
    round_idx = 0

    while not sim.done:
        candidates = sim.candidates()
        if not candidates:
            break
        env = sim.selector_env()
        device_states = sim.device_states()
        # The scope guard wants the admitted set; use every device
        # currently in the sim's state map.
        admitted = list(device_states.keys())
        ranked = policy.rank_contacts(
            candidates,
            device_states,
            env,
            pass_kind=MissionPass.COLLECT,
            admitted=admitted,
        )
        if not ranked:
            # Policy declared nothing feasible. End the mission.
            break
        chosen = ranked[0]
        result = sim.step(chosen)
        rounds.append(Exp3RoundLog(
            round_index=round_idx,
            n_updates=result.completed_count,
            n_target=result.member_count,
            deadline_met=True,  # contact was served before the mission deadline
        ))
        round_idx += 1

    # Pass-2 — count how many devices the greedy walk would reach with
    # whatever budget remains. Each remaining contact adds its devices
    # to the reach count if it fits in the residual budget.
    pass2_reached = 0
    for contact in sim.candidates():
        # Conservative cost: transit + collect (no upload — Pass 2 is a
        # downlink + ACK, much smaller than upload).
        from .sim_env import _euclid

        transit = _euclid(sim.mule_pose, contact.position) / cfg.sim.cruise_speed_m_s
        cost = transit + cfg.sim.session_time_s
        if cost > sim.budget_remaining:
            break
        pass2_reached += len(contact.devices)
    sim.record_pass2_deliveries(pass2_reached)

    summary = summarise_trial(
        rounds=rounds,
        metrics=sim.episode_metrics,
        cal=cal,
        n_devices=n_devices,
        is_mule_arm=True,
    )
    return summary
