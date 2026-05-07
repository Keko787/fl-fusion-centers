"""EX-3.3 — Exp3Sim sim env tests.

Pins the contracts the experiments plan §4.2 calls out:

* 3 base stations along the field edge (default).
* Time-varying per-channel upload rates depend on contact-to-BS distance.
* Fixed-speed mule: ``transit_s = dist / cruise_speed_m_s``.
* Configurable N / β / deadline_het / rrf affect the produced episode.
"""

from __future__ import annotations

import pytest

from experiments.exp3.sim_env import Exp3Sim, Exp3SimConfig


def _basic_cfg(**overrides):
    base = dict(
        n_devices=8, beta=1.0, deadline_heterogeneity=False,
        rf_range_m=60.0, world_radius=100.0,
        cruise_speed_m_s=10.0,
        mission_budget_s=600.0,
        seed=0,
    )
    base.update(overrides)
    return Exp3SimConfig(**base)


def test_default_has_three_base_stations():
    sim = Exp3Sim(_basic_cfg())
    assert len(sim.base_station_positions) == 3


def test_reset_partitions_devices_into_contacts():
    sim = Exp3Sim(_basic_cfg(n_devices=10, rf_range_m=300.0))
    sim.reset()
    contacts = sim.candidates()
    assert sum(len(c.devices) for c in contacts) == 10


def test_step_uses_fixed_speed_mule():
    sim = Exp3Sim(_basic_cfg(cruise_speed_m_s=5.0, n_devices=4, rf_range_m=300.0))
    sim.reset()
    contacts = sim.candidates()
    assert len(contacts) == 1
    chosen = contacts[0]
    result = sim.step(chosen)
    # transit_s should equal dist / cruise_speed_m_s within float precision.
    expected = result.transit_distance_m / 5.0
    assert result.transit_s == pytest.approx(expected, rel=1e-9)


def test_upload_rate_depends_on_distance_to_nearest_bs():
    sim = Exp3Sim(_basic_cfg(seed=42))
    sim.reset()
    # Sample the rate at a position very close to a BS vs far away.
    bs = sim.base_station_positions[0]
    near_rate = sim._upload_rate_at((bs[0], bs[1], 0.0))
    far_rate = sim._upload_rate_at((-10000.0, -10000.0, 0.0))
    # The far position is clamped via the floor (0.4·nominal); the near
    # position gets the full nominal rate (× a small Gaussian jitter).
    # Average over a few draws to make the assertion robust to jitter.
    near_samples = [sim._upload_rate_at((bs[0], bs[1], 0.0)) for _ in range(10)]
    far_samples = [sim._upload_rate_at((-10000.0, -10000.0, 0.0)) for _ in range(10)]
    assert sum(near_samples) / 10 > sum(far_samples) / 10


def test_deadline_heterogeneity_creates_two_groups():
    sim = Exp3Sim(_basic_cfg(deadline_heterogeneity=True, n_devices=10))
    sim.reset()
    windows = sorted({
        sim.device_states()[did].deadline_fulfilment_s
        for did in sim.device_states()
    })
    # Two distinct deadline windows when heterogeneity is on.
    assert len(windows) == 2


def test_deadline_uniform_when_heterogeneity_off():
    sim = Exp3Sim(_basic_cfg(deadline_heterogeneity=False, n_devices=10))
    sim.reset()
    windows = sorted({
        sim.device_states()[did].deadline_fulfilment_s
        for did in sim.device_states()
    })
    assert len(windows) == 1


def test_step_records_per_device_completion_counters():
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0, seed=0))
    sim.reset()
    chosen = sim.candidates()[0]
    sim.step(chosen)
    visits = sim.episode_metrics.per_device_visits
    assert sum(visits.values()) == 4  # all 4 served in parallel


def test_done_when_budget_exhausted():
    sim = Exp3Sim(_basic_cfg(mission_budget_s=10.0, n_devices=10, beta=1.0))
    sim.reset()
    # Budget = 10s but session_time alone is 30s → done immediately after
    # construction (or no contact fits).
    assert sim.done


def test_record_pass2_deliveries_validates_input():
    sim = Exp3Sim(_basic_cfg())
    sim.reset()
    sim.record_pass2_deliveries(3)
    assert sim.episode_metrics.pass2_devices_reached == 3
    with pytest.raises(ValueError):
        sim.record_pass2_deliveries(-1)


def test_step_advances_now_and_consumes_budget():
    sim = Exp3Sim(_basic_cfg(n_devices=4, rf_range_m=300.0, mission_budget_s=300.0))
    sim.reset()
    before_budget = sim.budget_remaining
    before_now = sim.now
    sim.step(sim.candidates()[0])
    assert sim.budget_remaining < before_budget
    assert sim.now > before_now


def test_selector_env_carries_mission_deadline_and_rate_hook():
    sim = Exp3Sim(_basic_cfg(beta=2.0, mission_budget_s=100.0))
    sim.reset()
    env = sim.selector_env()
    assert env.mission_deadline_s == pytest.approx(200.0)
    # The env's upload-rate hook returns a positive number at any pos.
    assert env.upload_rate_bps_at is not None
    assert env.upload_rate_bps_at((0.0, 0.0, 0.0)) > 0.0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(n_devices=0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(rf_range_m=-1.0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(cruise_speed_m_s=0.0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(beta=0.0))
    with pytest.raises(ValueError):
        Exp3Sim(Exp3SimConfig(base_station_positions=()))
