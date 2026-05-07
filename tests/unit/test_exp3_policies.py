"""EX-3.2 — A2 / A3 contact-ranking policy tests.

Pins the contracts the experiments plan §4.2 calls out:

* A2 (ArrivalOrderPolicy) keeps the input contact list in registration
  order; no skipping, no reordering.
* A3 (EdfFeasibilityPolicy) sorts by ``deadline_ts`` ascending and
  drops contacts whose feasibility cost exceeds the remaining budget.
* Both raise :class:`SelectorScopeViolation` in Pass 2 (the selector /
  policy surface is COLLECT-only).
* Both pass admitted-set and bucket-respecting scope checks.
"""

from __future__ import annotations

import pytest

from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
)
from hermes.scheduler.policies import ArrivalOrderPolicy, EdfFeasibilityPolicy
from hermes.scheduler.policies.edf_feasibility import FeasibilityModel
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.scope_guard import SelectorScopeViolation


def _make_contact(name: str, x: float, y: float, deadline: float) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, y, 0.0),
        devices=(DeviceID(name),),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=deadline,
    )


def _make_states(contacts):
    out = {}
    for c in contacts:
        for did in c.devices:
            st = DeviceSchedulerState(
                device_id=did,
                last_known_position=c.position,
            )
            st.bucket = c.bucket
            out[did] = st
    return out


# --------------------------------------------------------------------------- #
# A2 — ArrivalOrderPolicy
# --------------------------------------------------------------------------- #

def test_a2_keeps_registration_order():
    a = _make_contact("a", 0.0, 0.0, 100.0)
    b = _make_contact("b", 50.0, 50.0, 50.0)
    c = _make_contact("c", 99.0, 0.0, 999.0)
    contacts = [a, b, c]
    states = _make_states(contacts)

    pol = ArrivalOrderPolicy()
    out = pol.rank_contacts(contacts, states, SelectorEnv())
    assert out == contacts
    # Returns a copy — not the same list object.
    assert out is not contacts


def test_a2_empty_returns_empty():
    pol = ArrivalOrderPolicy()
    assert pol.rank_contacts([], {}, SelectorEnv()) == []


def test_a2_pass2_raises_scope_violation():
    pol = ArrivalOrderPolicy()
    a = _make_contact("a", 0.0, 0.0, 1.0)
    with pytest.raises(SelectorScopeViolation):
        pol.rank_contacts([a], _make_states([a]), SelectorEnv(),
                          pass_kind=MissionPass.DELIVER)


def test_a2_unadmitted_member_raises():
    a = _make_contact("a", 0.0, 0.0, 1.0)
    states = _make_states([a])
    pol = ArrivalOrderPolicy()
    with pytest.raises(SelectorScopeViolation):
        pol.rank_contacts([a], states, SelectorEnv(),
                          admitted=[DeviceID("not-a")])


# --------------------------------------------------------------------------- #
# A3 — EdfFeasibilityPolicy
# --------------------------------------------------------------------------- #

def test_a3_orders_earliest_deadline_first():
    """A3 must sort by deadline_ts ascending."""
    a = _make_contact("a", 0.0, 0.0, 200.0)
    b = _make_contact("b", 1.0, 0.0, 50.0)   # tightest
    c = _make_contact("c", 2.0, 0.0, 100.0)
    contacts = [a, b, c]
    states = _make_states(contacts)
    # Generous budget so the feasibility filter doesn't drop anyone.
    pol = EdfFeasibilityPolicy(model=FeasibilityModel(
        cruise_speed_m_s=100.0,  # essentially free transit
        session_time_s=1.0,
        base_station_positions=((0.0, 0.0, 0.0),),
        upload_bytes_per_contact=0.0,
        default_mission_budget_s=10_000.0,
    ))
    out = pol.rank_contacts(contacts, states, SelectorEnv())
    assert [c.devices[0] for c in out] == [
        DeviceID("b"), DeviceID("c"), DeviceID("a"),
    ]


def test_a3_drops_infeasible_when_budget_exhausted():
    """Two close-by + one far-away contact under a tight budget.

    The far-away contact should be dropped by the feasibility filter.
    """
    a = _make_contact("a", 5.0, 0.0, 100.0)
    b = _make_contact("b", 10.0, 0.0, 110.0)
    far = _make_contact("far", 1000.0, 0.0, 999.0)
    contacts = [a, b, far]
    states = _make_states(contacts)
    pol = EdfFeasibilityPolicy(model=FeasibilityModel(
        cruise_speed_m_s=1.0,        # 1 m/s — far is 1000 s away
        session_time_s=5.0,
        base_station_positions=((0.0, 0.0, 0.0),),
        upload_bytes_per_contact=0.0,
        default_mission_budget_s=200.0,
    ))
    out = pol.rank_contacts(contacts, states, SelectorEnv())
    member_ids = [c.devices[0] for c in out]
    assert DeviceID("far") not in member_ids
    assert DeviceID("a") in member_ids


def test_a3_pass2_raises_scope_violation():
    pol = EdfFeasibilityPolicy()
    a = _make_contact("a", 0.0, 0.0, 1.0)
    with pytest.raises(SelectorScopeViolation):
        pol.rank_contacts([a], _make_states([a]), SelectorEnv(),
                          pass_kind=MissionPass.DELIVER)


def test_a3_empty_returns_empty():
    pol = EdfFeasibilityPolicy()
    assert pol.rank_contacts([], {}, SelectorEnv()) == []


def test_a3_zero_remaining_budget_returns_empty():
    """If env.now ≥ mission_deadline, A3 returns []."""
    a = _make_contact("a", 0.0, 0.0, 1.0)
    states = _make_states([a])
    # mission_deadline_s = 100 via getattr; env.now = 200 ⇒ remaining < 0.
    from experiments.exp3.sim_env import Exp3SelectorEnv

    env = Exp3SelectorEnv(now=200.0, mission_deadline_s=100.0)
    pol = EdfFeasibilityPolicy()
    assert pol.rank_contacts([a], states, env) == []


def test_a3_uses_env_upload_rate_callable():
    """When the env exposes ``upload_rate_bps_at``, A3 should consult it."""
    a = _make_contact("a", 100.0, 0.0, 1.0)
    states = _make_states([a])
    # Tight budget; upload_bytes large; rate from env hits 0.5 bps.
    from experiments.exp3.sim_env import Exp3SelectorEnv

    env = Exp3SelectorEnv(
        now=0.0,
        mission_deadline_s=10.0,
        upload_rate_bps_at=lambda pos: 0.5,  # absurdly slow
    )
    pol = EdfFeasibilityPolicy(model=FeasibilityModel(
        cruise_speed_m_s=1000.0,  # transit ~ 0
        session_time_s=1.0,
        upload_bytes_per_contact=1e6,  # 8 Mbits at 0.5 bps ⇒ ages
        default_mission_budget_s=10.0,
    ))
    # The contact should be infeasible because upload_s > budget.
    assert pol.rank_contacts([a], states, env) == []
