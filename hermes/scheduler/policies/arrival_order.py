"""A2 — Arrival-order policy.

Services contacts in the registration order produced upstream by S3a's
clustering. No skipping, no reordering — captures the paper's "what if
the scheduler does nothing?" baseline.

The policy is stateful only across a single ``rank_contacts`` call: the
returned list is the input list, copied (so the caller can safely mutate
it without disturbing future calls). Tied with ``TargetSelectorRL``'s
:meth:`rank_contacts` API surface so the Experiment-3 driver can swap
A2 / A3 / A4 through the same constructor slot.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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


class ArrivalOrderPolicy:
    """Returns contacts in the order S3a handed them to us.

    The S3a clusterer's output is the canonical "arrival" sequence — it
    maps directly to the registration timeline (devices that registered
    first cluster first because they sit at the head of the
    ``eligible_device_ids`` list). Preserving that order is exactly the
    "no scheduling" baseline the paper asks for.
    """

    name = "A2"

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
                f"ArrivalOrderPolicy.rank_contacts called with "
                f"pass_kind={pass_kind.value!r}; A2 is a Pass-1-only policy."
            )
        if not candidates:
            return []
        members: List[DeviceID] = []
        for wp in candidates:
            members.extend(wp.devices)
        assert_candidates_admitted(
            members, admitted if admitted is not None else members,
        )
        return list(candidates)
