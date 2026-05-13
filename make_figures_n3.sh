#!/usr/bin/env bash
# make_figures_n3.sh
#
# Regenerate every figure for the N=3 experiment set in
# DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md once all six experiments
# have finished. Reads each run's artifacts under results/ and writes
# .png plots into results/figures/.
#
# Run from the project root with the venv active:
#     source .venv/bin/activate
#     bash make_figures_n3.sh
#
# Prerequisites — these run dirs must already exist (created by the
# experiment commands in §2 of the runbook with --epochs 5
# --rounds 50 --num_clients 3 except where noted):
#
#   results/exp1_centralized/               # exp 1 (centralized, --epochs 250)
#   results/exp3_fedavg_iid_n3/              # exp 3
#   results/exp4_fedavg_geo_n3/              # exp 4
#   results/exp5_fedprox_geo_n3/             # exp 5
#   results/exp6_scaling_n3/                 # exp 6 N=3 multi-node
#   results/exp6_scaling_n5/                 # exp 6 N=5 simulation
#   results/exp6_scaling_n10/                # exp 6 N=10 simulation
#
# Missing exp 6 sub-runs are skipped (scaling figure won't render
# without all three N values). Every other figure errors loudly if its
# input is missing.

set -euo pipefail

OUT_DIR="results/figures"
mkdir -p "$OUT_DIR"

# ── Centralized baseline ─────────────────────────────────────────────
# Pull the final-epoch threat_macro_f1 from exp 1's eval log to use as
# the horizontal reference line on the headline plot.
BASELINE_LOG=$(ls -1 results/exp1_centralized/*_evaluation.log 2>/dev/null | tail -1 || true)
if [[ -z "${BASELINE_LOG}" ]]; then
    echo "ERROR: no evaluation log found under results/exp1_centralized/"
    echo "       Run experiment 1 first."
    exit 1
fi
BASELINE=$(grep "^threat_macro_f1:" "$BASELINE_LOG" | tail -1 | awk '{print $2}')
echo "Centralized baseline macro-F1 = $BASELINE  (from $BASELINE_LOG)"

# ── §4.1 per-client class distribution ───────────────────────────────
echo "[1/5] per-client class distribution..."
python -m Analysis.CommunitiesCrime.plot_per_client_distribution \
    --run_dir results/exp4_fedavg_geo_n3 \
    --output "$OUT_DIR/per_client_distribution_geo_n3.png"

# ── §4.2 headline convergence ────────────────────────────────────────
echo "[2/5] headline convergence (centralized vs federated)..."
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedAvg-IID-n3=results/exp3_fedavg_iid_n3 \
           FedAvg-geo-n3=results/exp4_fedavg_geo_n3 \
           FedProx-geo-n3=results/exp5_fedprox_geo_n3 \
    --centralized-baseline "$BASELINE" \
    --output "$OUT_DIR/headline_n3.png"

# ── §4.3 scaling sweep ───────────────────────────────────────────────
if [[ -d results/exp6_scaling_n3 && -d results/exp6_scaling_n5 && -d results/exp6_scaling_n10 ]]; then
    echo "[3/5] scaling sweep (N=3, 5, 10)..."
    python -m Analysis.CommunitiesCrime.plot_scaling_n_clients \
        --runs 3=results/exp6_scaling_n3 \
               5=results/exp6_scaling_n5 \
               10=results/exp6_scaling_n10 \
        --output "$OUT_DIR/scaling_n_clients.png"
else
    echo "[3/5] scaling sweep SKIPPED — missing one of: exp6_scaling_n{3,5,10}"
fi

# ── §4.4 proximal-term evolution ─────────────────────────────────────
echo "[4/5] FedProx proximal-term evolution..."
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedProx-geo-n3=results/exp5_fedprox_geo_n3 \
    --metric proximal_contribution \
    --output "$OUT_DIR/proximal_evolution_n3.png"

# ── §4.4 fairness variance ───────────────────────────────────────────
echo "[5/5] per-client fairness variance..."
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedAvg-geo-n3=results/exp4_fedavg_geo_n3 \
           FedProx-geo-n3=results/exp5_fedprox_geo_n3 \
    --metric fairness_accuracy_variance \
    --output "$OUT_DIR/fairness_n3.png"

echo ""
echo "Done. Figures written to $OUT_DIR/"
ls -la "$OUT_DIR"
