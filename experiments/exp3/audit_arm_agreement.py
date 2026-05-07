"""Diagnostic: do A2 / A3 / A4 actually make different decisions?

Why this exists
---------------

The paper-grade Experiment 3b sweep produces the headline result that
A2/A3/A4 are mutually indistinguishable on every federation-side
metric. Before publishing that as a real finding (rather than a bug),
we need to verify that the three arms are actually exercising
different policies — i.e. that they disagree on which contact to
visit at each decision point.

Three failure modes this script flags:

1. **Decision count too low.** If a typical mission only makes 2 or 3
   ranking decisions before the mission budget runs out, the strategies
   have very little room to differ. This is a property of the simulator
   parameters, not a bug, but it is the most likely explanation of a
   null result.

2. **A3's feasibility filter never triggers.** ``EdfFeasibilityPolicy``
   reduces to "EDF over the candidate list" if the filter never drops
   anything. We log filter activations to detect this.

3. **A4 picks identically to A2 or A3.** If the trained selector's
   argmax is consistently the same as arrival-order or earliest-deadline,
   then either the trained policy is degenerate, or the candidate list
   is too short for ranking to matter. Either way, the paper text
   should disclose it.

Usage
-----

    python -m experiments.exp3.audit_arm_agreement \\
        --selector-weights weights/a4_selector.npz \\
        --seeds 20 \\
        --n-devices 10 --beta 1.0 --rrf 60.0

The script prints per-seed decision sequences and an aggregate report
covering decision counts, pairwise agreement rates, and A3 filter
activation. No CSV is written — the output is for human inspection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from hermes.scheduler.policies import (
    ArrivalOrderPolicy,
    EdfFeasibilityPolicy,
)
from hermes.scheduler.policies.edf_feasibility import FeasibilityModel
from hermes.scheduler.selector import TargetSelectorRL
from hermes.scheduler.selector.ddqn import DDQN
from hermes.types import ContactWaypoint, MissionPass

from .sim_env import Exp3Sim, Exp3SimConfig


log = logging.getLogger("experiments.exp3.audit_arm_agreement")


# --------------------------------------------------------------------------- #
# Per-arm trial: run one mission, record the chosen sequence
# --------------------------------------------------------------------------- #

def _contact_key(c: ContactWaypoint) -> Tuple[str, ...]:
    """Stable identity for a contact across arms — its device tuple."""
    return tuple(sorted(str(d) for d in c.devices))


def _run_one_arm(
    *,
    arm_name: str,
    policy,
    cfg: Exp3SimConfig,
    track_filter_drops: bool,
) -> Tuple[List[Tuple[str, ...]], List[int]]:
    """Run one mission end-to-end with the given policy.

    Returns (chosen_sequence, drops_per_decision).

    ``chosen_sequence`` is the list of contact identities (device tuples)
    visited in order; ``drops_per_decision`` is the count of candidates
    removed by the policy at each decision (always zero for A2/A4;
    non-zero for A3 when its feasibility filter fires).
    """
    sim = Exp3Sim(cfg)
    sim.reset()

    chosen: List[Tuple[str, ...]] = []
    drops: List[int] = []

    while not sim.done:
        candidates = sim.candidates()
        if not candidates:
            break
        env = sim.selector_env()
        device_states = sim.device_states()
        admitted = list(device_states.keys())

        ranked = policy.rank_contacts(
            candidates, device_states, env,
            pass_kind=MissionPass.COLLECT, admitted=admitted,
        )
        if track_filter_drops:
            drops.append(max(0, len(candidates) - len(ranked)))
        else:
            drops.append(0)

        if not ranked:
            break
        head = ranked[0]
        chosen.append(_contact_key(head))
        sim.step(head)

    return chosen, drops


# --------------------------------------------------------------------------- #
# Aggregate: pairwise agreement + decision counts + A3 filter rate
# --------------------------------------------------------------------------- #

def _pairwise_agreement(
    seq_a: List[Tuple[str, ...]],
    seq_b: List[Tuple[str, ...]],
) -> Tuple[int, int]:
    """How many positions in the prefix agree, and the prefix length.

    Two sequences agree at position ``i`` if both have a contact there
    and they're equal. We compare prefixes up to the shorter of the two.
    Returns (agreements, prefix_length).
    """
    n = min(len(seq_a), len(seq_b))
    agreements = sum(1 for i in range(n) if seq_a[i] == seq_b[i])
    return agreements, n


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp3.audit_arm_agreement")
    parser.add_argument(
        "--selector-weights", type=Path, default=None,
        help="Path to a trained DDQN .npz from experiments.exp3.train_a4. "
             "Required for an honest A4 audit; if omitted, A4 uses random "
             "init and the audit is meaningless for that arm.",
    )
    parser.add_argument(
        "--seeds", type=int, default=20,
        help="Number of paired-seed trials to audit (default 20).",
    )
    parser.add_argument("--n-devices", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.25,
                        help="Default 0.25 puts the audit in the "
                             "budget-tight regime where the filter "
                             "has a chance to fire.")
    parser.add_argument("--rrf", type=float, default=60.0)
    parser.add_argument("--mission-budget-s", type=float, default=600.0)
    parser.add_argument("--cruise-speed-m-s", type=float, default=5.0)
    parser.add_argument(
        "--upload-bytes", type=float, default=1.0e7,
        help="Per-contact upload payload (default 10 MB).",
    )
    parser.add_argument(
        "--upload-bps", type=float, default=1.0e6,
        help="Nominal upload rate (default 1 Mbps).",
    )
    parser.add_argument(
        "--jittery", action="store_true",
        help=(
            "Activate Exp.\\ 1-parity network-link noise: 2%% packet "
            "loss + 30%% latency jitter on every per-device "
            "transmission and per-step duration. Off by default."
        ),
    )
    parser.add_argument(
        "--packet-loss-pct", type=float, default=0.0,
        help="Override the per-device packet-loss percentage. "
             "Ignored if --jittery is on (which forces 2%%).",
    )
    parser.add_argument(
        "--latency-jitter-pct", type=float, default=0.0,
        help="Override the per-step latency-jitter percentage. "
             "Ignored if --jittery is on (which forces 30%%).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every trial's chosen-contact sequence per arm.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # Build the three arms.
    if args.selector_weights is None:
        log.warning(
            "no --selector-weights given; A4 will use a RANDOM-INIT DDQN. "
            "Re-run with the trained weights for an honest audit."
        )
        a4_selector = TargetSelectorRL(epsilon=0.0)
    else:
        ddqn = DDQN.load(args.selector_weights)
        a4_selector = TargetSelectorRL(ddqn=ddqn, epsilon=0.0)
        log.info("loaded A4 selector weights from %s", args.selector_weights)

    arms = [
        ("A2", ArrivalOrderPolicy(), False),
        (
            "A3",
            EdfFeasibilityPolicy(model=FeasibilityModel(
                cruise_speed_m_s=args.cruise_speed_m_s,
                session_time_s=30.0,
                upload_bytes_per_contact=args.upload_bytes,
                nominal_upload_bps=args.upload_bps,
                default_mission_budget_s=args.mission_budget_s * args.beta,
            )),
            True,  # track filter drops
        ),
        ("A4", a4_selector, False),
    ]

    # Per-arm bookkeeping across seeds.
    sequences_by_arm: dict = {name: [] for name, _, _ in arms}
    drops_by_arm: dict = {name: [] for name, _, _ in arms}

    # Run paired seeds.
    for seed_idx in range(args.seeds):
        # Jittery mode forces Exp.\ 1-parity noise (2% loss, 30% jit).
        # Otherwise honour explicit overrides.
        if args.jittery:
            packet_loss = 2.0
            latency_jitter = 30.0
        else:
            packet_loss = float(args.packet_loss_pct)
            latency_jitter = float(args.latency_jitter_pct)
        cfg = Exp3SimConfig(
            n_devices=args.n_devices,
            beta=args.beta,
            rf_range_m=args.rrf,
            mission_budget_s=args.mission_budget_s,
            cruise_speed_m_s=args.cruise_speed_m_s,
            upload_bytes_per_contact=args.upload_bytes,
            nominal_upload_bps=args.upload_bps,
            packet_loss_pct=packet_loss,
            latency_jitter_pct=latency_jitter,
            seed=1000 + seed_idx,
        )
        if args.verbose:
            print(f"--- seed {1000 + seed_idx} ---")
        for arm_name, policy, track_drops in arms:
            seq, drops = _run_one_arm(
                arm_name=arm_name, policy=policy,
                cfg=cfg, track_filter_drops=track_drops,
            )
            sequences_by_arm[arm_name].append(seq)
            drops_by_arm[arm_name].append(drops)
            if args.verbose:
                seq_str = " -> ".join("/".join(c) for c in seq) or "(none)"
                drop_str = (
                    f"  filter drops/decision: {drops}"
                    if track_drops else ""
                )
                print(f"  {arm_name}: {len(seq)} contacts | {seq_str}{drop_str}")

    # ------------------------------------------------------------------ #
    # Aggregate report.
    # ------------------------------------------------------------------ #
    print()
    print("=" * 64)
    print("AUDIT REPORT")
    print("=" * 64)

    # 1. Decision counts per trial.
    print("\n[1] Decisions per trial (mean ± std):")
    for arm_name in ("A2", "A3", "A4"):
        counts = [len(s) for s in sequences_by_arm[arm_name]]
        if not counts:
            print(f"    {arm_name}: no trials recorded")
            continue
        mean = sum(counts) / len(counts)
        var = sum((c - mean) ** 2 for c in counts) / max(1, len(counts) - 1)
        std = var ** 0.5
        print(
            f"    {arm_name}: {mean:.2f} ± {std:.2f}  "
            f"(min={min(counts)}, max={max(counts)})"
        )

    # 2. Pairwise sequence-prefix agreement.
    print("\n[2] Pairwise decision agreement (prefix-aligned, paired seeds):")
    pairs = [("A2", "A3"), ("A2", "A4"), ("A3", "A4")]
    for a, b in pairs:
        total_agree = 0
        total_compare = 0
        identical_sequences = 0
        for i in range(args.seeds):
            seq_a = sequences_by_arm[a][i]
            seq_b = sequences_by_arm[b][i]
            agree, n_compare = _pairwise_agreement(seq_a, seq_b)
            total_agree += agree
            total_compare += n_compare
            if seq_a == seq_b:
                identical_sequences += 1
        if total_compare == 0:
            rate_str = "n/a"
        else:
            rate_str = f"{total_agree / total_compare:.1%}"
        print(
            f"    {a} vs {b}: {rate_str} of {total_compare} aligned decisions "
            f"agree | {identical_sequences}/{args.seeds} trials produced "
            f"identical sequences"
        )

    # 3. A3 feasibility-filter trigger rate.
    print("\n[3] A3 feasibility filter:")
    a3_drops = drops_by_arm["A3"]
    total_decisions = sum(len(d) for d in a3_drops)
    decisions_with_drop = sum(
        1 for d in a3_drops for n in d if n > 0
    )
    total_dropped = sum(n for d in a3_drops for n in d)
    if total_decisions == 0:
        print("    no decisions logged")
    else:
        print(
            f"    fired on {decisions_with_drop} / {total_decisions} "
            f"decisions ({decisions_with_drop / total_decisions:.1%})"
        )
        print(
            f"    total candidates dropped: {total_dropped}"
        )
        if decisions_with_drop == 0:
            print(
                "    NOTE: A3 reduced to plain EDF (no skipping) for "
                "every decision. This is consistent with a budget-rich "
                "regime where every contact is feasible."
            )

    # 4. Verdict heuristic.
    print("\n[4] Verdict heuristic:")
    counts_a2 = [len(s) for s in sequences_by_arm["A2"]]
    mean_decisions = (
        sum(counts_a2) / len(counts_a2) if counts_a2 else 0.0
    )
    a4_a3_identical = sum(
        1 for i in range(args.seeds)
        if sequences_by_arm["A4"][i] == sequences_by_arm["A3"][i]
    )
    a4_a2_identical = sum(
        1 for i in range(args.seeds)
        if sequences_by_arm["A4"][i] == sequences_by_arm["A2"][i]
    )

    if mean_decisions < 3:
        print(
            "    Mean decisions per trial < 3. The simulator is "
            "budget-bound; ordering 2-3 contacts can't differentiate "
            "ranking strategies. Null result is most likely a property "
            "of the parameter regime, not a bug."
        )
    elif a4_a3_identical >= 0.9 * args.seeds:
        print(
            "    A4 produced identical sequences to A3 in 90%+ of "
            "trials. Either the trained policy is degenerate or the "
            "candidate list is too short to differentiate strategies. "
            "Re-train A4 with different hyperparameters or increase "
            "the simulator's decision pressure (larger N, slower "
            "cruise, longer budget)."
        )
    elif a4_a2_identical >= 0.9 * args.seeds:
        print(
            "    A4 produced identical sequences to A2 in 90%+ of "
            "trials. The trained policy is approximating arrival "
            "order; very likely a degenerate Q-distribution. "
            "Re-train A4."
        )
    else:
        print(
            "    A4 makes different decisions but produces equivalent "
            "metrics. Honest finding: ranking matters but the metrics "
            "are insensitive to ordering at this budget. Disclose in "
            "paper text rather than chase a non-existent bug."
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
