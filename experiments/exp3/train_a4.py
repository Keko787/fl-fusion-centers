"""CLI to train the A4 selector and write its weights to disk.

Wraps :func:`hermes.scheduler.selector.selector_train.train_selector_contact`
with an argparse interface + an ``.npz`` save step. Produces the file
:mod:`experiments.exp3.runner_main` consumes via ``--selector-weights``.

Usage::

    python -m experiments.exp3.train_a4 \\
        --episodes 400 \\
        --rf-range-m 60 \\
        --output weights/a4_selector.npz

The trainer is :func:`train_selector_contact`, which runs DQN training
against :class:`~hermes.scheduler.selector.sim_env.ContactSim`. That sim
shares the contact-event reward shape with :class:`Exp3Sim`, so weights
trained against ``ContactSim`` transfer directly to the experiment's
evaluation environment without retraining (per implementation plan §4.2
"preserve reward shape so retraining is optional").
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from hermes.scheduler.selector.selector_train import (
    ContactTrainConfig,
    run_ab_evaluation_contact,
    train_selector_contact,
)


log = logging.getLogger("experiments.exp3.train_a4")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp3.train_a4")
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the trained DDQN weights (.npz).",
    )
    parser.add_argument(
        "--episodes", type=int, default=400,
        help="Training episodes (default: 400).",
    )
    parser.add_argument(
        "--n-devices", type=int, default=8,
        help="Devices per training episode bucket (default: 8).",
    )
    parser.add_argument(
        "--rf-range-m", type=float, default=60.0,
        help="S3a clustering radius for training (default: 60.0).",
    )
    parser.add_argument(
        "--mission-budget", type=float, default=200.0,
        help="Per-episode mission budget seconds (default: 200.0).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
    )
    parser.add_argument(
        "--warmup", type=int, default=64,
        help="Replay-buffer warmup before SGD starts (default: 64).",
    )
    parser.add_argument(
        "--epsilon-start", type=float, default=0.9,
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=0.05,
    )
    parser.add_argument(
        "--epsilon-decay-episodes", type=int, default=300,
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--ab-episodes", type=int, default=0,
        help=(
            "Run an A/B vs the distance baseline after training and print "
            "the multi-metric scoreboard. 0 = skip (default)."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cfg = ContactTrainConfig(
        episodes=args.episodes,
        n_devices=args.n_devices,
        rf_range_m=args.rf_range_m,
        mission_budget=args.mission_budget,
        batch_size=args.batch_size,
        warmup=args.warmup,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        seed=args.seed,
    )

    log.info(
        "training A4 selector: episodes=%d n_devices=%d rf_range_m=%.1f seed=%d",
        cfg.episodes, cfg.n_devices, cfg.rf_range_m, cfg.seed,
    )
    selector, metrics = train_selector_contact(cfg=cfg)

    if metrics.mean_reward_by_episode:
        first = sum(metrics.mean_reward_by_episode[:20]) / min(
            20, len(metrics.mean_reward_by_episode)
        )
        last = sum(metrics.mean_reward_by_episode[-20:]) / min(
            20, len(metrics.mean_reward_by_episode)
        )
        log.info(
            "training done: %d episodes, first-20 mean reward %+.3f, "
            "last-20 mean reward %+.3f, %d SGD steps",
            cfg.episodes, first, last, len(metrics.loss_by_step),
        )

    if args.ab_episodes > 0:
        log.info("running A/B against distance baseline (%d episodes)", args.ab_episodes)
        ab = run_ab_evaluation_contact(
            selector,
            episodes=args.ab_episodes,
            n_devices=cfg.n_devices,
            rf_range_m=cfg.rf_range_m,
            mission_budget=cfg.mission_budget,
            seed=cfg.seed + 1,
        )
        log.info(
            "A/B: completion_lift=%+.2f%%  energy_savings=%+.2f%%  "
            "retry_savings=%+.2f%%  compute_overhead=%.1fx",
            100.0 * ab.completion_rate_lift,
            100.0 * ab.energy_savings,
            100.0 * ab.retry_rate_savings,
            ab.compute_overhead_x,
        )

    selector.ddqn.save(args.output)
    log.info("wrote weights to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
