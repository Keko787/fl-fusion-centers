"""Experiment-3 CLI entry point.

Walks the A1×A2×A3×A4 grid via the EX-0 :class:`TrialRunner` and writes
a per-trial CSV. Resumable across sessions via the CSV log's
already-done set.

Usage::

    python -m experiments.exp3.runner_main \\
        --csv results/exp3.csv \\
        --n-trials 20

The grid axes are the four sweep knobs from §IV-D plus the four arms
(A1, A2, A3, A4). The defaults mirror the implementation plan §4.2
sweep — ``N ∈ {5, 10, 20}``, ``β ∈ {0.5, 1.0, 2.0}``,
``deadline_het ∈ {False, True}``, ``rrf ∈ {30, 60, 120}``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from experiments.runner import TrialGrid, TrialRunner

from hermes.scheduler.selector import TargetSelectorRL
from hermes.scheduler.selector.ddqn import DDQN

from .driver import ARMS, Exp3Driver
from .metrics import Exp3MetricSummary


log = logging.getLogger("experiments.exp3.runner_main")


def _build_grid(
    *,
    arms: Sequence[str],
    Ns: Sequence[int],
    betas: Sequence[float],
    rrfs: Sequence[float],
    deadline_het_modes: Sequence[bool],
    n_trials: int,
    base_seed: int = 42,
) -> TrialGrid:
    return TrialGrid(
        independent_vars={
            "N": list(Ns),
            "beta": list(betas),
            "rrf": list(rrfs),
            "deadline_het": list(deadline_het_modes),
        },
        arms=list(arms),
        n_trials=n_trials,
        base_seed=base_seed,
    )


def _metric_columns() -> list[str]:
    """Metric + extra param columns the CSV declares.

    The runner adds ``cell_id``, ``arm``, ``trial_index``, ``seed``,
    and ``param_*`` automatically; we own the rest.
    """
    return Exp3MetricSummary.csv_columns() + [
        "n_devices",
        "beta",
        "deadline_het",
        "rf_range_m",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp3.runner_main")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Per-trial CSV path (created if missing).")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Trials per cell (paired across arms).")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        help="Which arms to run; subset of A1 A2 A3 A4.")
    parser.add_argument("--N", nargs="+", type=int, default=[5, 10, 20],
                        help="Bucket-size sweep.")
    parser.add_argument("--beta", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                        help="Deadline-tightness sweep.")
    parser.add_argument("--rrf", nargs="+", type=float, default=[30.0, 60.0, 120.0],
                        help="rf_range_m sweep.")
    parser.add_argument("--deadline-het", nargs="+", type=int,
                        default=[0, 1],
                        help="0 = uniform deadlines; 1 = heterogeneous.")
    parser.add_argument("--timeout-s", type=float, default=300.0,
                        help="Soft per-trial timeout (warning only).")
    parser.add_argument(
        "--selector-weights", type=Path, default=None,
        help=(
            "Path to a trained DDQN .npz produced by "
            "experiments.exp3.train_a4. If omitted, A4 uses a "
            "RANDOM-INIT network (smoke-test only — not paper-grade)."
        ),
    )
    parser.add_argument(
        "--require-trained-a4", action="store_true",
        help=(
            "Refuse to run when A4 would use random-init weights. "
            "Set this on every paper-grade run."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    grid = _build_grid(
        arms=args.arms,
        Ns=args.N,
        betas=args.beta,
        rrfs=args.rrf,
        deadline_het_modes=[bool(b) for b in args.deadline_het],
        n_trials=args.n_trials,
        base_seed=args.base_seed,
    )

    selector_a4: Optional[TargetSelectorRL] = None
    if args.selector_weights is not None:
        ddqn = DDQN.load(args.selector_weights)
        selector_a4 = TargetSelectorRL(ddqn=ddqn, epsilon=0.0)
        log.info("loaded A4 selector weights from %s", args.selector_weights)
    elif "A4" in args.arms:
        msg = (
            "A4 is in --arms but --selector-weights was not provided. "
            "A4 will use random-init DDQN weights — results are not "
            "paper-grade. Train weights first with "
            "`python -m experiments.exp3.train_a4 --output PATH` and "
            "re-run with --selector-weights PATH."
        )
        if args.require_trained_a4:
            parser.error(msg)
        log.warning(msg)

    driver = Exp3Driver(selector_a4=selector_a4)
    runner = TrialRunner(
        grid=grid,
        log_path=args.csv,
        metric_columns=_metric_columns(),
        timeout_s=args.timeout_s,
    )
    n = runner.run(driver.run_trial)
    print(f"wrote {n} new trial rows to {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
