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

    # Figures 0a-0d — all-arms comparison, one figure per metric.
    # Shows A1/A2/A3/A4 side by side so the progressive-sophistication
    # story (no scheduling -> arrival -> EDF -> RL) is visible per
    # metric. Each figure carries the metric's definition + direction
    # of improvement so the reader doesn't need to cross-reference the
    # paper text.
    arms_order = [a for a in ("A1", "A2", "A3", "A4") if a in arms]
    # (fig_id, attribute, y-axis label) — titles/captions intentionally
    # omitted from the rendered figures; they belong in the LaTeX caption
    # produced by ``write_latex_captions`` so the paper figure environment
    # owns the prose layer.
    metric_figs = [
        ("fig0a", "update_yield", "Updates aggregated per round (mean)"),
        ("fig0b", "round_close_rate_kminhalf",
         "Fraction of rounds closed (k_min = N/2)"),
        ("fig0c", "jains_fairness", "Jain's fairness index"),
        ("fig0d", "coverage", "Fraction of scheduled devices serviced"),
    ]
    for fig_id, metric, ylabel in metric_figs:
        try:
            if len(arms_order) < 2:
                continue
            data: List[List[float]] = []
            labels: List[str] = []
            for arm in arms_order:
                vals = [
                    getattr(r, metric) for r in rows
                    if r.is_ok and r.arm == arm
                    and getattr(r, metric) is not None
                ]
                if vals:
                    data.append(vals)
                    labels.append(arm)
            if not data:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(
                data, labels=labels, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 6},
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Arm")
            ax.grid(True, axis="y", alpha=0.3)
            # Vertical separator between A1 (centralized FL upper bound)
            # and the mule-arm relay-FL ablation (A2/A3/A4). The metric
            # is not directly comparable across the two groupings — A1's
            # "round" parallelises clients, while the mule arms' "round"
            # is one contact event — so the separator visually flags
            # that the boxplot mixes a control with an ablation set.
            if labels and labels[0] == "A1" and len(labels) > 1:
                ax.axvline(x=1.5, color="#888888", linestyle="--",
                           linewidth=1.4, alpha=0.85, zorder=0)
                # Extend the y-axis upward so the region labels sit in
                # fresh whitespace instead of crashing into A1's high
                # outliers. 18% headroom is enough that label boxes
                # sit comfortably above the topmost data point.
                ymin, ymax = ax.get_ylim()
                yspan_orig = ymax - ymin
                ax.set_ylim(ymin, ymax + 0.18 * yspan_orig)
                ymin, ymax = ax.get_ylim()
                label_y = ymin + 0.97 * (ymax - ymin)
                ax.text(
                    1.0, label_y,
                    "Centralized FL\n(upper bound)",
                    ha="center", va="top", fontsize=8.5,
                    style="italic", color="#444444",
                )
                ax.text(
                    (len(labels) + 2) / 2.0, label_y,
                    "Mule relay-FL ablation",
                    ha="center", va="top", fontsize=8.5,
                    style="italic", color="#444444",
                )
            _watermark(ax)
            fig.tight_layout()
            out = figures_dir / f"exp3_{fig_id}_{metric}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
        except Exception as e:  # pragma: no cover
            log.warning("%s (%s) skipped: %s", fig_id, metric, e)

    # Figure 0e — mule-only propulsion energy. A1 has no mule, so this
    # is a separate three-arm comparison (A2/A3/A4) on the Eq. 5 cost
    # ledger.
    try:
        mule_arms = [a for a in ("A2", "A3", "A4") if a in arms]
        if len(mule_arms) >= 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            data: List[List[float]] = []
            labels: List[str] = []
            for arm in mule_arms:
                vals = [
                    r.propulsion_energy_J for r in rows
                    if r.is_ok and r.arm == arm
                    and r.propulsion_energy_J is not None
                ]
                if vals:
                    data.append(vals)
                    labels.append(arm)
            if data:
                ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_ylabel("Propulsion energy per mission (J)")
            ax.set_xlabel("Arm")
            ax.grid(True, axis="y", alpha=0.3)
            _watermark(ax)
            fig.tight_layout()
            out = figures_dir / "exp3_fig0e_propulsion_energy.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    except Exception as e:  # pragma: no cover
        log.warning("fig0e (propulsion energy) skipped: %s", e)

    # Emit LaTeX caption blocks for every metric figure produced above.
    try:
        latex_path = _write_latex_captions(figures_dir)
        written.append(latex_path)
    except Exception as e:  # pragma: no cover
        log.warning("LaTeX captions skipped: %s", e)

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

    # Figure 4 — β-sweep, all N values overlaid on one panel.
    # Mule-arm ablation only (A1 omitted): A1 is the centralized-FL
    # upper bound and is shown alongside A2/A3/A4 in fig0a; including
    # it here would dominate the y-axis and obscure the finer
    # scheduling-strategy comparison among A2/A3/A4. fig0a already
    # establishes the upper-bound magnitude; this figure focuses on the
    # mule-arm scheduling ablation.
    try:
        Ns = sorted({r.N for r in rows if r.is_ok})
        if Ns and arms:
            # Stable color per arm so identity carries across figures.
            # A1 entry retained for future re-inclusion if needed.
            arm_colors = {
                "A1": "tab:blue", "A2": "tab:orange",
                "A3": "tab:green", "A4": "tab:red",
            }
            # Line style + marker per N (more N values cycle through
            # the lists; three is the typical case).
            n_styles = ["-", "--", ":"]
            n_markers = ["o", "s", "^"]

            fig, ax = plt.subplots(figsize=(7.5, 5))
            arms_in_data = [a for a in ("A2", "A3", "A4") if a in arms]
            for arm in arms_in_data:
                color = arm_colors.get(arm, "black")
                for i, N in enumerate(Ns):
                    style = n_styles[i % len(n_styles)]
                    marker = n_markers[i % len(n_markers)]
                    betas = sorted({r.beta for r in rows
                                    if r.is_ok and r.N == N})
                    ys: List[float] = []
                    for b in betas:
                        cell = [r for r in rows
                                if r.is_ok and r.arm == arm
                                and r.N == N and r.beta == b]
                        ys.append(np.mean([r.update_yield for r in cell])
                                  if cell else float("nan"))
                    ax.plot(
                        betas, ys,
                        color=color, linestyle=style, marker=marker,
                        markersize=6, linewidth=1.8,
                    )
            ax.set_xlabel("β (deadline tightness)")
            ax.set_ylabel("Update yield (mean)")
            ax.grid(True, alpha=0.3)

            # Two legends: arm (color) and N (line style). Drawn as
            # proxy artists so they don't clutter the data lines.
            from matplotlib.lines import Line2D
            arm_handles = [
                Line2D([0], [0], color=arm_colors.get(a, "black"),
                       linewidth=2, label=a)
                for a in arms_in_data
            ]
            n_handles = [
                Line2D([0], [0], color="black",
                       linestyle=n_styles[i % len(n_styles)],
                       marker=n_markers[i % len(n_markers)],
                       linewidth=1.8, markersize=6,
                       label=f"N = {N}")
                for i, N in enumerate(Ns)
            ]
            # With A1 omitted, the mule-arm lines sit in the upper
            # half of the rescaled y-axis (≈ 1.0–2.2 yield). Drop the
            # Arm legend to the lower-left so it doesn't crash into
            # the dotted N=20 line at the top, and keep the
            # bucket-size legend at upper-right where the band above
            # the dotted line is clear.
            arm_legend = ax.legend(
                handles=arm_handles, title="Arm",
                loc="lower left", fontsize=9, title_fontsize=9,
            )
            ax.add_artist(arm_legend)
            ax.legend(
                handles=n_handles, title="Bucket size",
                loc="upper right", fontsize=9, title_fontsize=9,
            )
            # Annotate the two distinct regimes the chart contains. A1
            # (blue) is a centralized-FL upper bound — its yield reflects
            # per-round client parallelism with no mule-transit
            # constraint, so it is not directly comparable to A2/A3/A4
            # on a per-round basis. The mule arms cluster at the bottom
            # of the chart and constitute the relay-FL scheduling
            # ablation. Text annotations make this distinction explicit
            # on the figure itself rather than only in the caption.
            _watermark(ax)
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
# LaTeX caption emitter — pairs with the bare PNG figures
# --------------------------------------------------------------------------- #

# (file_stem, label_id, short_caption, full_caption_body)
# Kept module-level so the test suite + the paper toolchain can import
# them without re-running the figure generation.
_LATEX_CAPTIONS: tuple = (
    (
        "exp3_fig0a_update_yield",
        "fig:exp3:update_yield",
        "Update yield across arms",
        (
            r"\textbf{Update yield} per round across A1 (centralized FL), "
            r"A2 (arrival-order), A3 (EDF heuristic), A4 (RL). "
            r"Higher is better. Update yield is the mean count of "
            r"client updates aggregated per round; it captures how "
            r"effectively each scheduling strategy converts a round "
            r"of mission time into committed FL contributions."
        ),
    ),
    (
        "exp3_fig0b_round_close_rate_kminhalf",
        "fig:exp3:close_rate",
        "Round close rate (k\\textsubscript{min} = N/2) across arms",
        (
            r"\textbf{Round close rate} at $k_{\min} = N/2$ across the "
            r"four arms. Higher is better. A round closes when at "
            r"least $k_{\min}$ aggregated updates arrive within "
            r"deadline; the metric thus rewards consistency rather "
            r"than averaging high-yield and empty rounds."
        ),
    ),
    (
        "exp3_fig0c_jains_fairness",
        "fig:exp3:fairness",
        "Jain's fairness across arms",
        (
            r"\textbf{Jain's fairness index} on per-device service "
            r"counts, $J = (\sum_i x_i)^2 / (N \cdot \sum_i x_i^2)$, "
            r"range $[1/N, 1]$. Higher is better; $J = 1$ is "
            r"perfectly equal service. The metric exposes whether a "
            r"scheduling strategy concentrates service on a small "
            r"subset of devices or spreads it across the slice."
        ),
    ),
    (
        "exp3_fig0d_coverage",
        "fig:exp3:coverage",
        "Coverage across arms",
        (
            r"\textbf{Coverage} across the four arms, defined as the "
            r"fraction of scheduled devices serviced at least once "
            r"during the mission. Higher is better; range $[0, 1]$. "
            r"Coverage is the binary complement of the per-device "
            r"miss rate and is independent of how many times a "
            r"device was served beyond the first visit."
        ),
    ),
    (
        "exp3_fig0e_propulsion_energy",
        "fig:exp3:propulsion_energy",
        "Mule propulsion energy across mule arms",
        (
            r"\textbf{Propulsion energy per mission} (joules) for the "
            r"three mule arms (A1 has no mule and is therefore "
            r"omitted). Lower is better. Computed via Eq.~5: "
            r"$E = T_{\text{mission}} \cdot P_{\text{idle}} + "
            r"B_{\text{tx}} \cdot \varepsilon_{\text{bit}} + "
            r"L_{\text{path}} \cdot \varepsilon_{\text{prop}}$. "
            r"The three mule arms produce overlapping distributions in "
            r"the studied parameter range; the energy ledger does not "
            r"discriminate among A2, A3, and A4."
        ),
    ),
    (
        "exp3_fig4_beta_sweep",
        "fig:exp3:beta_sweep",
        "Mule-arm update yield versus deadline tightness, across bucket sizes",
        (
            r"\textbf{Mule-arm update yield versus deadline tightness "
            r"$\beta$}, with bucket size $N$ encoded by line style "
            r"(solid: $N=5$; dashed: $N=10$; dotted: $N=20$) and arm "
            r"encoded by colour. The centralized-FL upper bound (A1) "
            r"is omitted from this figure to bring the y-axis to a "
            r"scale where the mule-arm comparison is legible; the "
            r"upper-bound magnitude is established in "
            r"Fig.~\ref{fig:exp3:update_yield}. The three mule arms "
            r"(A2/A3/A4) overlap at every $(N, \beta)$ cell, scaling "
            r"with bucket size but insensitive to deadline slack. "
            r"This $\beta$-insensitivity isolates mule travel time as "
            r"the binding constraint on contacts-per-mission: "
            r"additional deadline slack provides no measurable gain "
            r"once the mule has exhausted its mission budget on "
            r"transit between contacts."
        ),
    ),
)


def _write_latex_captions(figures_dir: Path) -> Path:
    """Emit a ``.tex`` snippet with one ``\\begin{figure}`` per metric.

    The figures themselves carry no titles or footer text — all the
    explanation lives in the LaTeX caption produced here, which the
    paper can ``\\input{}`` directly.
    """
    out = figures_dir / "exp3_fig_captions.tex"
    figures_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "% Auto-generated by experiments.analysis.exp3.write_figures().",
        "% One \\begin{figure} block per metric figure produced alongside",
        "% this file. The PNGs carry no titles or footers; all caption",
        "% text lives here so the paper owns the prose layer.",
        "",
    ]
    for stem, label, short, full in _LATEX_CAPTIONS:
        lines.extend([
            r"\begin{figure}[t]",
            r"  \centering",
            f"  \\includegraphics[width=0.7\\linewidth]{{figures/exp3/{stem}.png}}",
            f"  \\caption[{short}]{{{full}}}",
            f"  \\label{{{label}}}",
            r"\end{figure}",
            "",
        ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


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
