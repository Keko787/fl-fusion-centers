"""A3 — Earliest-deadline-first with feasibility skip.

Implements the paper's heuristic baseline: each ranking step picks the
contact with the soonest deadline, then drops any contact whose
estimated cost (transit + collect + return + upload) exceeds the
remaining mission budget. Drops are silent — the caller (Experiment-3
driver) sees a shorter list than it handed in.

Feasibility cost model::

    transit_s  = dist(mule_pose, contact.position)            / cruise_speed_m_s
    collect_s  = session_time_s                              (per contact)
    return_s   = dist(contact.position, nearest_bs_pose)      / cruise_speed_m_s
    upload_s   = upload_bytes / upload_rate_bps_at_position * 8

A contact is **infeasible** when
``transit + collect + return + upload > remaining_budget_s``. The
remaining-budget term comes from :class:`SelectorEnv` (the supervisor
populates ``env.now`` and the experiment driver passes the deadline as
``mission_deadline_s``).

The policy intentionally has no learning component — that's A4's job.
The *only* knobs are physical-cost constants the experiment driver
either reads from the calibration TOML or fixes in the trial harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from hermes.types import (
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
)

from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.scope_guard import (
    SelectorScopeViolation,
    assert_candidates_admitted,
)


Position = Tuple[float, float, float]


@dataclass(frozen=True)
class FeasibilityModel:
    """Cost model the EDF policy uses to test feasibility.

    Defaults match the values in :mod:`hermes.scheduler.selector.sim_env`
    so a stand-alone EdfFeasibilityPolicy run produces sensible numbers
    against ContactSim — the experiment driver overrides them with the
    extended-sim's calibrated values.
    """

    cruise_speed_m_s: float = 5.0
    session_time_s: float = 30.0
    base_station_positions: Tuple[Position, ...] = ((0.0, 0.0, 0.0),)
    # Bytes uploaded per contact; the paper sweeps this via the
    # |D|pd × bucket-size product, but the policy itself only needs a
    # single scalar per call.
    upload_bytes_per_contact: float = 0.0
    # Time-varying upload rate is read from the env's current position
    # via ``rate_at(position)``; if None, falls back to the constant
    # below (so a default-constructed model still works in tests).
    nominal_upload_bps: float = 1.0e7  # 10 Mbps default
    # Mission budget in seconds — when env carries it, prefer that;
    # otherwise this hard cap is used.
    default_mission_budget_s: float = float("inf")


def _euclid(a: Position, b: Position) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _nearest_bs(position: Position, bses: Sequence[Position]) -> Position:
    if not bses:
        return position
    return min(bses, key=lambda b: _euclid(position, b))


class EdfFeasibilityPolicy:
    """Earliest-deadline-first ranking with cost-based feasibility skip.

    Construction takes a :class:`FeasibilityModel`; ``rank_contacts``
    is otherwise pure (no per-call state besides the immutable model).
    """

    name = "A3"

    def __init__(self, model: Optional[FeasibilityModel] = None) -> None:
        self._model = model if model is not None else FeasibilityModel()

    @property
    def model(self) -> FeasibilityModel:
        return self._model

    def rank_contacts(
        self,
        candidates: Sequence[ContactWaypoint],
        device_states: Dict[DeviceID, DeviceSchedulerState],
        env: SelectorEnv,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted: Optional[Sequence[DeviceID]] = None,
    ) -> List[ContactWaypoint]:
        if pass_kind is not MissionPass.COLLECT:
            raise SelectorScopeViolation(
                f"EdfFeasibilityPolicy.rank_contacts called with "
                f"pass_kind={pass_kind.value!r}; A3 is a Pass-1-only policy."
            )
        if not candidates:
            return []

        members: List[DeviceID] = []
        for wp in candidates:
            members.extend(wp.devices)
        assert_candidates_admitted(
            members, admitted if admitted is not None else members,
        )

        # Determine the remaining-budget envelope the policy is allowed
        # to spend. The selector env carries ``now``; the experiment
        # driver injects the absolute mission deadline by stuffing it on
        # the env via the ``mission_deadline_s`` attribute — use getattr
        # so a vanilla SelectorEnv (no extra attribute) still works.
        mission_deadline = float(
            getattr(env, "mission_deadline_s", self._model.default_mission_budget_s)
        )
        remaining = mission_deadline - float(env.now)
        if remaining <= 0.0:
            return []

        # 1. EDF ordering on the deadline_ts of each contact (tighter
        #    deadline first). Tie-break on stable position+devices for
        #    determinism across re-runs.
        ordered = sorted(
            candidates,
            key=lambda c: (c.deadline_ts, c.position, c.devices),
        )

        # 2. Greedy feasibility filter. The mule "advances" through the
        #    ordered queue; each kept contact updates the simulated mule
        #    pose so subsequent transit estimates are correct.
        kept: List[ContactWaypoint] = []
        pose: Position = env.mule_pose
        budget = remaining

        for wp in ordered:
            transit = _euclid(pose, wp.position) / max(
                self._model.cruise_speed_m_s, 1e-6
            )
            collect = self._model.session_time_s
            return_dist = _euclid(
                wp.position,
                _nearest_bs(wp.position, self._model.base_station_positions),
            )
            return_s = return_dist / max(self._model.cruise_speed_m_s, 1e-6)
            upload_rate = self._upload_rate_at(wp.position, env)
            upload = (
                self._model.upload_bytes_per_contact * 8.0 / max(upload_rate, 1e-6)
            )
            cost = transit + collect + return_s + upload
            if cost > budget:
                continue
            kept.append(wp)
            budget -= cost
            pose = wp.position

        return kept

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _upload_rate_at(self, position: Position, env: SelectorEnv) -> float:
        """Resolve the per-position upload rate.

        Honours an env-attached ``upload_rate_bps_at`` callable for
        time-varying rates (Experiment-3 sim env exposes this). Falls
        back to the model's nominal rate otherwise — keeps the policy
        usable from unit tests without standing up the extended sim.
        """
        rate_fn = getattr(env, "upload_rate_bps_at", None)
        if callable(rate_fn):
            try:
                return float(rate_fn(position))
            except Exception:
                # Defensive: sim-side bugs shouldn't crash the policy.
                pass
        return float(self._model.nominal_upload_bps)
