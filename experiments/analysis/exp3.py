"""Experiment-3 analysis — load CSV → stats + paper figures.

Reads the per-trial CSV produced by :mod:`experiments.exp3.runner_main`
and emits the 6 paper figures (per implementation plan §4.2 EX-3.5)
plus a one-page text summary.

Runnable directly::

    python -m experiments.analysis.exp3 \\
        --csv results/exp3.csv \\
        --figures-dir figures/exp3/

The six figures + one CSV companion:

1. A2 vs A1 — paired Wilcoxon on update yield + Jain's fairness.
2. A3 vs A2 — paired Wilcoxon on round close rate + propulsion energy.
3. A4 vs A3 — paired Wilcoxon on update yield + round close rate
   (the experiment's primary novelty).
4. β-sweep curve: update yield vs β with one curve per arm,
   faceted by N. The slope-vs-cliff figure.
5. rrf-sweep curve: update yield vs rrf with one curve per arm at
   ``β=1.0, N=10``.
6. ρ_contact bar chart faceted by rrf, comparing A2/A3/A4.

Plus ``exp3_paired_tests.csv`` for paper-table reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments.calibration import load_calibration

from .stats import (
    PairedTestResult,
    paired_wilcoxon_with_cliffs_delta,
)

log = logging.getLogger("experiments.analysis.exp3")


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Exp3Row:
    """One row of the Experiment-3 CSV after type coercion."""

    cell_id: str
    arm: str
    trial_index: int
    seed: int
    N: int
    beta: float
    deadline_het: bool
    rf_range_m: float
    update_yield: float
    coverage: float
    jains_fairness: float
    participation_entropy: float
    round_close_rate_kmin1: float
    round_close_rate_kminhalf: float
    round_close_rate_kminN: float
    rho_contact: Optional[float]
    pass2_coverage: Optional[float]
    propulsion_energy_J: Optional[float]
    propulsion_idle_J: Optional[float]
    propulsion_tx_J: Optional[float]
    propulsion_prop_J: Optional[float]
    mission_completion_s: Optional[float]
    path_length_m: Optional[float]
    status: str

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"


def _opt_float(s: Any) -> Optional[float]:
    if s in (None, "", "None"):
        return None
    return float(s)


def _opt_bool(s: Any) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).lower() in ("true", "1", "yes")


def load_trials(csv_path: Path) -> List[Exp3Row]:
    csv_path = Path(csv_path)
    rows: List[Exp3Row] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "cell_id", "arm", "trial_index", "seed",
            "param_N", "param_beta", "param_rrf", "param_deadline_het",
            "update_yield", "coverage", "jains_fairness",
            "participation_entropy",
            "round_close_rate_kmin1", "round_close_rate_kminhalf",
            "round_close_rate_kminN",
            "rho_contact", "pass2_coverage",
            "propulsion_energy_J", "mission_completion_s", "path_length_m",
            "status",
        }
        cols = set(reader.fieldnames or [])
        missing = required - cols
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {sorted(missing)}"
            )
        for raw in reader:
            try:
                rows.append(Exp3Row(
                    cell_id=raw["cell_id"],
                    arm=raw["arm"],
                    trial_index=int(raw["trial_index"]),
                    seed=int(raw["seed"]),
                    N=int(raw["param_N"]),
                    beta=float(raw["param_beta"]),
                    deadline_het=_opt_bool(raw["param_deadline_het"]),
                    rf_range_m=float(raw["param_rrf"]),
                    update_yield=float(raw["update_yield"] or 0.0),
                    coverage=float(raw["coverage"] or 0.0),
                    jains_fairness=float(raw["jains_fairness"] or 0.0),
                    participation_entropy=float(
                        raw["participation_entropy"] or 0.0
                    ),
                    round_close_rate_kmin1=float(
                        raw["round_close_rate_kmin1"] or 0.0
                    ),
                    round_close_rate_kminhalf=float(
                        raw["round_close_rate_kminhalf"] or 0.0
                    ),
                    round_close_rate_kminN=float(
                        raw["round_close_rate_kminN"] or 0.0
                    ),
                    rho_contact=_opt_float(raw["rho_contact"]),
                    pass2_coverage=_opt_float(raw["pass2_coverage"]),
                    propulsion_energy_J=_opt_float(raw["propulsion_energy_J"]),
                    propulsion_idle_J=_opt_float(
                        raw.get("propulsion_idle_J", "")
                    ),
                    propulsion_tx_J=_opt_float(
                        raw.get("propulsion_tx_J", "")
                    ),
                    propulsion_prop_J=_opt_float(
                        raw.get("propulsion_prop_J", "")
                    ),
                    mission_completion_s=_opt_float(
                        raw["mission_completion_s"]
                    ),
                    path_length_m=_opt_float(raw["path_length_m"]),
                    status=raw["status"],
                ))
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"failed to parse row {raw!r}: {e}"
                ) from e
    return rows


# --------------------------------------------------------------------------- #
# Paired-test helpers
# --------------------------------------------------------------------------- #

def _pairs(
    rows: Sequence[Exp3Row], arm_a: str, arm_b: str, metric: str,
) -> Tuple[List[float], List[float]]:
    """Pair (cell_id, trial_index) trials across two arms; return aligned arrays."""
    a_by = {(r.cell_id, r.trial_index): r for r in rows if r.is_ok and r.arm == arm_a}
    b_by = {(r.cell_id, r.trial_index): r for r in rows if r.is_ok and r.arm == arm_b}
    a_vals: List[float] = []
    b_vals: List[float] = []
    for k, ar in a_by.items():
        br = b_by.get(k)
        if br is None:
            continue
        av = getattr(ar, metric)
        bv = getattr(br, metric)
        if av is None or bv is None:
            continue
        a_vals.append(float(av))
        b_vals.append(float(bv))
    return a_vals, b_vals


@dataclass(frozen=True)
class ArmPairTest:
    arm_a: str
    arm_b: str
    metric: str
    n_pairs: int
    test: Optional[PairedTestResult]


def paired_test(
    rows: Sequence[Exp3Row], arm_a: str, arm_b: str, metric: str,
) -> ArmPairTest:
    a, b = _pairs(rows, arm_a, arm_b, metric)
    if len(a) < 2:
        return ArmPairTest(arm_a, arm_b, metric, len(a), None)
    return ArmPairTest(
        arm_a, arm_b, metric, len(a),
        paired_wilcoxon_with_cliffs_delta(a, b),
    )


# --------------------------------------------------------------------------- #
# Text summary
# --------------------------------------------------------------------------- #

def summarize(rows: Sequence[Exp3Row]) -> str:
    if not rows:
        return "(no rows)"
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.is_ok)
    n_err = sum(1 for r in rows if r.status == "error")
    arms = sorted({r.arm for r in rows})
    lines = [
        f"Experiment 3 — summary (n={n_total} trials, ok={n_ok}, err={n_err})",
        f"  arms: {arms}",
        "",
    ]

    pairings = (
        ("A2", "A1", "update_yield", "A2 vs A1 (slow-deadline claim)"),
        ("A2", "A1", "jains_fairness", "A2 vs A1 fairness"),
        ("A3", "A2", "round_close_rate_kminhalf", "A3 vs A2 close-rate"),
        ("A3", "A2", "propulsion_energy_J", "A3 vs A2 mission energy"),
        ("A4", "A3", "update_yield", "A4 vs A3 update yield"),
        ("A4", "A3", "round_close_rate_kminhalf", "A4 vs A3 close-rate"),
    )
    for arm_a, arm_b, metric, label in pairings:
        result = paired_test(rows, arm_a, arm_b, metric)
        if result.test is None:
            lines.append(
                f"  {label}: (insufficient pairs — n={result.n_pairs})"
            )
            continue
        t = result.test
        sig = "*" if t.significant else " "
        lines.append(
            f"  {label}: n={result.n_pairs} W={t.statistic:.1f} "
            f"p={t.p_value:.4f}{sig} δ={t.cliffs_delta:+.3f} "
            f"({t.delta_magnitude})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def write_figures(
    rows: Sequence[Exp3Row],
    *,
    figures_dir: Path,
    placeholder_watermark: bool = False,
) -> List[Path]:
    """Emit the six paper figures + paired-tests CSV.

    Returns the list of written paths. Each figure is wrapped in a
    try/except so a degenerate input (e.g. a CSV with only one arm)
    doesn't abort the whole batch.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def _watermark(ax) -> None:
        if not placeholder_watermark:
            return
        ax.text(
            0.5, 0.5, "PLACEHOLDER CALIBRATION",
            transform=ax.transAxes, fontsize=20, color="red",
            alpha=0.25, ha="center", va="center", rotation=30,
        )

    arms = sorted({r.arm for r in rows if r.is_ok})

    # 1 + 2 + 3 — paired tests CSV (three rows: A2vsA1, A3vsA2, A4vsA3).
    try:
        sig_path = figures_dir / "exp3_paired_tests.csv"
        with open(sig_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "comparison", "metric", "n_pairs",
                "W", "p", "cliffs_delta", "delta_magnitude",
            ])
            comparisons = (
                ("A2", "A1", "update_yield"),
                ("A2", "A1", "jains_fairness"),
                ("A3", "A2", "round_close_rate_kminhalf"),
                ("A3", "A2", "propulsion_energy_J"),
                ("A4", "A3", "update_yield"),
                ("A4", "A3", "round_close_rate_kminhalf"),
            )
            for arm_a, arm_b, metric in comparisons:
                pt = paired_test(rows, arm_a, arm_b, metric)
                if pt.test is None:
                    w.writerow([f"{arm_a}_vs_{arm_b}", metric, pt.n_pairs,
                                "", "", "", ""])
                    continue
                w.writerow([
                    f"{arm_a}_vs_{arm_b}", metric, pt.n_pairs,
                    pt.test.statistic, pt.test.p_value,
                    pt.test.cliffs_delta, pt.test.delta_magnitude,
                ])
        written.append(sig_path)
    except Exception as e:  # pragma: no cover
        log.warning("paired tests CSV skipped: %s", e)

    # Figure 1 — A2 vs A1 paired comparison panel (yield + fairness).
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(
            axes,
            ("update_yield", "jains_fairness"),
            ("Update yield", "Jain's fairness"),
        ):
            a, b = _pairs(rows, "A2", "A1", metric)
            if a and b:
                ax.boxplot([b, a], labels=["A1", "A2"], showmeans=True)
            ax.set_ylabel(title)
            ax.set_title(f"{title}: A2 vs A1")
            _watermark(ax)
        fig.suptitle("A2 vs A1 (slow-deadline claim)")
        fig.tight_layout()
        out = figures_dir / "exp3_fig1_a2_vs_a1.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig1 skipped: %s", e)

    # Figure 2 — A3 vs A2 (close rate + propulsion energy).
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(
            axes,
            ("round_close_rate_kminhalf", "propulsion_energy_J"),
            ("Round close rate (kmin=N/2)", "Propulsion energy (J)"),
        ):
            a, b = _pairs(rows, "A3", "A2", metric)
            if a and b:
                ax.boxplot([b, a], labels=["A2", "A3"], showmeans=True)
            ax.set_ylabel(title)
            ax.set_title(f"{title}: A3 vs A2")
            _watermark(ax)
        fig.suptitle("A3 vs A2 (heuristic EDF gain)")
        fig.tight_layout()
        out = figures_dir / "exp3_fig2_a3_vs_a2.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig2 skipped: %s", e)

    # Figure 3 — A4 vs A3 (yield + close rate; primary novelty).
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(
            axes,
            ("update_yield", "round_close_rate_kminhalf"),
            ("Update yield", "Round close rate (kmin=N/2)"),
        ):
            a, b = _pairs(rows, "A4", "A3", metric)
            if a and b:
                ax.boxplot([b, a], labels=["A3", "A4"], showmeans=True)
            ax.set_ylabel(title)
            ax.set_title(f"{title}: A4 vs A3")
            _watermark(ax)
        fig.suptitle("A4 vs A3 (HERMES RL primary novelty)")
        fig.tight_layout()
        out = figures_dir / "exp3_fig3_a4_vs_a3.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig3 skipped: %s", e)

    # Figure 4 — β-sweep curve, faceted by N.
    try:
        Ns = sorted({r.N for r in rows if r.is_ok})
        if Ns and arms:
            fig, axes = plt.subplots(1, len(Ns), figsize=(4 * len(Ns), 4),
                                     sharey=True)
            if len(Ns) == 1:
                axes = [axes]
            for ax, N in zip(axes, Ns):
                betas = sorted({r.beta for r in rows if r.is_ok and r.N == N})
                for arm in arms:
                    ys: List[float] = []
                    for b in betas:
                        cell = [r for r in rows
                                if r.is_ok and r.arm == arm
                                and r.N == N and r.beta == b]
                        ys.append(np.mean([r.update_yield for r in cell])
                                  if cell else float("nan"))
                    ax.plot(betas, ys, marker="o", label=arm)
                ax.set_xlabel("β (deadline tightness)")
                ax.set_ylabel("Update yield")
                ax.set_title(f"N = {N}")
                ax.legend(fontsize=8)
                _watermark(ax)
            fig.suptitle("Update yield vs β (slope-vs-cliff)")
            fig.tight_layout()
            out = figures_dir / "exp3_fig4_beta_sweep.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig4 skipped: %s", e)

    # Figure 5 — rrf-sweep curve at β=1.0, N=10.
    try:
        target_rows = [
            r for r in rows
            if r.is_ok and abs(r.beta - 1.0) < 1e-9 and r.N == 10
        ]
        if target_rows:
            rrfs = sorted({r.rf_range_m for r in target_rows})
            fig, ax = plt.subplots(figsize=(7, 4))
            for arm in arms:
                ys = []
                for rrf in rrfs:
                    cell = [r for r in target_rows
                            if r.arm == arm and r.rf_range_m == rrf]
                    ys.append(np.mean([r.update_yield for r in cell])
                              if cell else float("nan"))
                ax.plot(rrfs, ys, marker="o", label=arm)
            ax.set_xlabel("rf_range_m (rrf)")
            ax.set_ylabel("Update yield")
            ax.set_title("Update yield vs rrf at β=1.0, N=10")
            ax.legend(fontsize=8)
            _watermark(ax)
            out = figures_dir / "exp3_fig5_rrf_sweep.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig5 skipped: %s", e)

    # Figure 6 — ρ_contact bar chart faceted by rrf, A2/A3/A4 only.
    try:
        mule_arms = [a for a in arms if a in ("A2", "A3", "A4")]
        rrfs = sorted({r.rf_range_m for r in rows if r.is_ok})
        if mule_arms and rrfs:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(rrfs))
            width = 0.8 / max(len(mule_arms), 1)
            for i, arm in enumerate(mule_arms):
                ys: List[float] = []
                for rrf in rrfs:
                    cell = [r for r in rows
                            if r.is_ok and r.arm == arm
                            and r.rf_range_m == rrf
                            and r.rho_contact is not None]
                    ys.append(
                        float(np.mean([r.rho_contact for r in cell]))
                        if cell else 0.0
                    )
                ax.bar(x + i * width, ys, width, label=arm)
            ax.set_xticks(x + width * (len(mule_arms) - 1) / 2)
            ax.set_xticklabels([f"{r:g}" for r in rrfs])
            ax.set_xlabel("rrf (m)")
            ax.set_ylabel("ρ_contact (mean devices/contact)")
            ax.set_title("ρ_contact across rrf for mule arms")
            ax.legend(fontsize=8)
            _watermark(ax)
            out = figures_dir / "exp3_fig6_rho_contact.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig6 skipped: %s", e)

    return written


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.analysis.exp3")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--figures-dir", default=Path("figures/exp3"), type=Path)
    parser.add_argument("--calibration", default=None, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cal = load_calibration(args.calibration)
    rows = load_trials(args.csv)
    text = summarize(rows)
    # Write via the raw stdout buffer so the Greek δ in the summary
    # survives Windows' default cp1252 console encoding.
    try:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    except AttributeError:  # pragma: no cover - non-binary stdout
        print(text)

    if not args.no_figures:
        figs = write_figures(
            rows,
            figures_dir=args.figures_dir,
            placeholder_watermark=not cal.is_paper_grade,
        )
        log.info("wrote %d figures/CSVs to %s", len(figs), args.figures_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
