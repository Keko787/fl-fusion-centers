#!/usr/bin/env bash
# make_figures_n3.sh
#
# Regenerate every figure for the N=3 experiment set in
# DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md once one or more
# experiments have finished. Reads each run's artifacts under results/
# and writes .png plots into results/figures/.
#
# Run from the project root with the venv active:
#     source .venv/bin/activate
#     bash make_figures_n3.sh
#
# Inputs (any subset is fine — each figure that can be rendered is
# rendered; the rest are skipped with a clearly-labeled warning):
#
#   results/exp1_centralized/               # exp 1 (centralized, --epochs 250)
#   results/exp3_fedavg_iid_n3/              # exp 3 — FedAvg / IID
#   results/exp4_fedavg_geo_n3/              # exp 4 — FedAvg / geographic
#   results/exp5_fedprox_geo_n3/             # exp 5 — FedProx / geographic
#   results/exp6_scaling_n3/                 # exp 6 N=3 multi-node
#   results/exp6_scaling_n5/                 # exp 6 N=5 simulation
#   results/exp6_scaling_n10/                # exp 6 N=10 simulation
#
# Safe to re-run as experiments finish — previously-written figures
# get overwritten in-place.

set -uo pipefail

OUT_DIR="results/figures"
mkdir -p "$OUT_DIR"

# A "ready" FL run dir has the server log written (i.e. at least round
# 1's aggregation completed). Used to gate plots on inputs being real.
fl_ready() { [[ -f "$1/server_evaluation.log" ]]; }

# A "ready" centralized run dir has at least one evaluation log.
central_ready() { ls -1 "$1"/*_evaluation.log >/dev/null 2>&1; }

# ── Centralized baseline (used by §4.2 headline) ─────────────────────
BASELINE=""
if central_ready results/exp1_centralized; then
    BASELINE_LOG=$(ls -1 results/exp1_centralized/*_evaluation.log | tail -1)
    BASELINE=$(grep "^threat_macro_f1:" "$BASELINE_LOG" | tail -1 | awk '{print $2}')
    echo "Centralized baseline macro-F1 = $BASELINE  (from $BASELINE_LOG)"
else
    echo "[warn] No evaluation log in results/exp1_centralized/ — headline plot will have no horizontal baseline."
fi

# ── §4.1 per-client class distribution ───────────────────────────────
if [[ -f results/exp4_fedavg_geo_n3/partition_stats.json ]]; then
    echo "[1/5] per-client class distribution..."
    python -m Analysis.CommunitiesCrime.plot_per_client_distribution \
        --run_dir results/exp4_fedavg_geo_n3 \
        --output "$OUT_DIR/per_client_distribution_geo_n3.png"
else
    echo "[1/5] per-client class distribution SKIPPED — missing results/exp4_fedavg_geo_n3/partition_stats.json (run exp 4)."
fi

# ── §4.2 headline convergence (any subset of exp3/4/5) ───────────────
HEADLINE_RUNS=()
fl_ready results/exp3_fedavg_iid_n3   && HEADLINE_RUNS+=("FedAvg-IID-n3=results/exp3_fedavg_iid_n3")
fl_ready results/exp4_fedavg_geo_n3   && HEADLINE_RUNS+=("FedAvg-geo-n3=results/exp4_fedavg_geo_n3")
fl_ready results/exp5_fedprox_geo_n3  && HEADLINE_RUNS+=("FedProx-geo-n3=results/exp5_fedprox_geo_n3")
if [[ ${#HEADLINE_RUNS[@]} -gt 0 ]]; then
    echo "[2/5] headline convergence (${#HEADLINE_RUNS[@]} run(s))..."
    HEADLINE_ARGS=()
    [[ -n "$BASELINE" ]] && HEADLINE_ARGS+=(--centralized-baseline "$BASELINE")
    python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
        --runs "${HEADLINE_RUNS[@]}" \
        "${HEADLINE_ARGS[@]}" \
        --output "$OUT_DIR/headline_n3.png"
else
    echo "[2/5] headline convergence SKIPPED — no FL run with a server_evaluation.log (run exp 3, 4, or 5)."
fi

# ── §4.3 scaling sweep (needs all three N values) ────────────────────
if fl_ready results/exp6_scaling_n3 && fl_ready results/exp6_scaling_n5 && fl_ready results/exp6_scaling_n10; then
    echo "[3/5] scaling sweep (N=3, 5, 10)..."
    python -m Analysis.CommunitiesCrime.plot_scaling_n_clients \
        --runs 3=results/exp6_scaling_n3 \
               5=results/exp6_scaling_n5 \
               10=results/exp6_scaling_n10 \
        --output "$OUT_DIR/scaling_n_clients.png"
else
    echo "[3/5] scaling sweep SKIPPED — missing one of: exp6_scaling_n{3,5,10}/server_evaluation.log (run exp 6 for all three N)."
fi

# ── §4.4 proximal-term evolution (FedProx only) ──────────────────────
if fl_ready results/exp5_fedprox_geo_n3; then
    echo "[4/5] FedProx proximal-term evolution..."
    python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
        --runs FedProx-geo-n3=results/exp5_fedprox_geo_n3 \
        --metric proximal_contribution \
        --output "$OUT_DIR/proximal_evolution_n3.png"
else
    echo "[4/5] proximal-term evolution SKIPPED — missing results/exp5_fedprox_geo_n3/server_evaluation.log (run exp 5)."
fi

# ── §4.4 fairness variance (FedAvg vs FedProx, geographic) ───────────
FAIRNESS_RUNS=()
fl_ready results/exp4_fedavg_geo_n3  && FAIRNESS_RUNS+=("FedAvg-geo-n3=results/exp4_fedavg_geo_n3")
fl_ready results/exp5_fedprox_geo_n3 && FAIRNESS_RUNS+=("FedProx-geo-n3=results/exp5_fedprox_geo_n3")
if [[ ${#FAIRNESS_RUNS[@]} -gt 0 ]]; then
    echo "[5/5] per-client fairness variance (${#FAIRNESS_RUNS[@]} run(s))..."
    python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
        --runs "${FAIRNESS_RUNS[@]}" \
        --metric fairness_accuracy_variance \
        --output "$OUT_DIR/fairness_n3.png"
else
    echo "[5/5] fairness variance SKIPPED — no FedAvg/FedProx geographic run with a server_evaluation.log."
fi

echo ""
echo "Done. Figures present in $OUT_DIR/:"
ls -la "$OUT_DIR" 2>/dev/null || echo "  (empty)"
