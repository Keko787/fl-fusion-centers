# Running the Fusion Centers FL Experiments

Phase E.5 quickstart — go from a fresh clone to the paper's headline figure. Companion to:
* [Fusion_Centers_FL_Update_Design.md](Fusion_Centers_FL_Update_Design.md)
* [Fusion_Centers_FL_Update_Implementation_Plan.md](Fusion_Centers_FL_Update_Implementation_Plan.md) — see **§9** for known limitations before the first real run.

The Phase 0–D work shipped a complete fusion-centers pipeline + 146-test suite. This document covers the **run-it-for-real** workflow: dataset acquisition, the six-experiment matrix, ablation passes, and figure regeneration.

---

## 0. Prerequisites

| Requirement | How |
|---|---|
| Python 3.9+ (project says 3.8+ but `argparse.BooleanOptionalAction` is 3.9) | `python --version` |
| TensorFlow 2.x + Flower 1.x + scikit-learn + scipy + matplotlib + joblib | `pip install tensorflow flwr scikit-learn scipy matplotlib joblib` |
| The UCI Communities and Crime (Unnormalized) archive | §1 below |

Chameleon Cloud (the design's deployment target) — see [README.md](../README.md) for AERPAW / Chameleon provisioning scripts. The fusion-centers experiments use the existing CPU-bare-metal pattern; outline §7.3 confirms compute is not the bottleneck.

### 0.1 One-shot deploy to a fresh node

For a clean machine (laptop, VM, Chameleon bare-metal, container), use the
fusion-centers bootstrap script under [`AppSetup/`](../AppSetup/). It
creates a local venv, installs [`requirements_fusion.txt`](../AppSetup/requirements_fusion.txt)
(a minimal pinned set, distinct from the legacy NIDS `requirements_core.txt`),
downloads the UCI archive into `$HOME/datasets/CommunitiesCrime/`,
generates the `.names` schema, and validates that all crime-rate columns
are present. Re-runs are idempotent.

```bash
# Linux / macOS / WSL
bash AppSetup/setup_fusion_node.sh            # install + dataset
bash AppSetup/setup_fusion_node.sh --verify   # also run pytest tests/unit -k fusion
```

```powershell
# Windows (PowerShell 7+)
pwsh -File AppSetup\setup_fusion_node.ps1
pwsh -File AppSetup\setup_fusion_node.ps1 -Verify
```

If you'd rather ship a container, build the slim image:

```bash
docker build -t fusion-centers -f AppSetup/DockerSetup/Dockerfile.fusion .
docker run --rm -v "$PWD/results:/app/results" fusion-centers \
    python App/TrainingApp/Client/TrainingClient.py \
        --model_type FUSION-MLP --trainingArea Central \
        --partition_strategy iid --num_clients 1 --client_id 0 \
        --epochs 50 --run_dir results/exp1_centralized \
        --save_name centralized_baseline
```

The image bakes the dataset in at build time and runs a schema smoke
check during the build, so every container starts from an identical,
validated state. If you'd rather mount the dataset, drop the `RUN curl`
block from the Dockerfile and pass `-v /host/datasets:/root/datasets`.

If you've used the bootstrap above, you can **skip §1 entirely** — the
script does it for you. §1 stays below for the manual-install case.

---

## 1. Data acquisition

The project expects the unnormalized UCI Communities and Crime archive under `$HOME/datasets/CommunitiesCrime/`. UCI 2.0 stopped shipping the `.names` schema file with the dataset zip — we generate it ourselves from the canonical metadata (see implementation plan §9.1).

```bash
# 1. Download the data file
mkdir -p ~/datasets/CommunitiesCrime
curl -L https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip \
    -o /tmp/comm_crime.zip
unzip -j /tmp/comm_crime.zip CommViolPredUnnormalizedData.txt -d ~/datasets/CommunitiesCrime/

# 2. Generate the .names schema via ucimlrepo (one-time)
pip install ucimlrepo
python -m Config.DatasetConfig.CommunitiesCrime_Sampling.generate_names_file
```

The generator script writes `~/datasets/CommunitiesCrime/communities_and_crime_unnormalized.names` (147 attributes) in the ARFF format `parse_names_file` expects.

**Validate the schema before any experiment run** — see implementation plan §9.1 + §9.2 for the URL/column-name + `.names` format risks. A 5-minute check (uses `$HOME/datasets/CommunitiesCrime/` by default; override `DATA_DIR` for a custom archive location):

```bash
DATA_DIR="${DATA_DIR:-$HOME/datasets/CommunitiesCrime}" \
python -c "
import os
from pathlib import Path
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    parse_names_file, load_raw, CRIME_RATE_COLUMNS, SENSITIVE_COLUMNS,
)
data_dir = Path(os.environ['DATA_DIR'])
names = parse_names_file(data_dir / 'communities_and_crime_unnormalized.names')
df = load_raw(data_dir / 'CommViolPredUnnormalizedData.txt', names)
print(f'rows={len(df)}, cols={df.shape[1]}')
missing_crime = [c for c in CRIME_RATE_COLUMNS if c not in df.columns]
missing_sensitive = [c for c in SENSITIVE_COLUMNS if c not in df.columns]
print(f'missing crime-rate cols: {missing_crime}')
print(f'missing sensitive cols (silently kept): {missing_sensitive}')
"
```

If any crime-rate column is missing, label engineering silently degrades. Adjust the constants in `commCrimeDatasetLoad.py` and re-run.

---

## 2. The six-experiment matrix

Each experiment lives in its own `--run_dir` under `results/`. Set the shared seed (`--commcrime_random_seed`) to the same value across all six so the global test split is identical.

### 2.1 Experiment 1 — Centralized baseline

```bash
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP \
    --trainingArea Central \
    --partition_strategy iid \
    --num_clients 1 \
    --client_id 0 \
    --epochs 50 \
    --run_dir results/exp1_centralized \
    --save_name centralized_baseline
```

Sets the macro-F1 ceiling. Phase B DoD threshold is `> 0.55` on the global test set. After this run, capture the number from `<run_dir>/<timestamp>_evaluation.log` for the `--centralized-baseline` argument to the headline plot.

### 2.2 Experiment 2 — Local-only (per client, no FL)

```bash
for cid in 0 1 2 3 4; do
  python App/TrainingApp/Client/TrainingClient.py \
      --model_type FUSION-MLP \
      --trainingArea Central \
      --partition_strategy geographic \
      --num_clients 5 \
      --client_id $cid \
      --epochs 50 \
      --run_dir results/exp2_local_only \
      --save_name local_only_n5_c${cid}
done
```

Lower-bound result — agencies operating in isolation. Use these client-by-client macro-F1 values to compute the "siloed performance" baseline.

### 2.3 Experiment 3 — FedAvg / IID

```bash
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP \
    --fl_strategy FedAvg \
    --partition_strategy iid \
    --num_clients 5 \
    --rounds 100 \
    --epochs 1 \
    --min_clients 5 \
    --run_dir results/exp3_fedavg_iid \
    --save_name fedavg_iid_n5
```

Sanity check — federation works in the easy case. Phase C DoD: macro-F1 within ±0.02 of centralized.

### 2.4 Experiment 4 — FedAvg / non-IID geographic

```bash
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP \
    --fl_strategy FedAvg \
    --partition_strategy geographic \
    --num_clients 5 \
    --rounds 100 \
    --epochs 1 \
    --min_clients 5 \
    --run_dir results/exp4_fedavg_geo \
    --save_name fedavg_geo_n5
```

Core result — realistic fusion-center scenario.

### 2.5 Experiment 5 — FedProx / non-IID geographic

```bash
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP \
    --fl_strategy FedProx --fedprox_mu 0.01 \
    --partition_strategy geographic \
    --num_clients 5 \
    --rounds 100 \
    --epochs 1 \
    --min_clients 5 \
    --run_dir results/exp5_fedprox_geo \
    --save_name fedprox_geo_n5
```

Robustness-to-drift claim — Phase D DoD: macro-F1 ≥ FedAvg on the same partition.

### 2.6 Experiment 6 — Scaling sweep

```bash
for n in 3 5 10; do
  python App/TrainingApp/HFLHost/HFLHost.py \
      --model_type FUSION-MLP \
      --fl_strategy FedAvg \
      --partition_strategy geographic \
      --num_clients $n \
      --rounds 100 \
      --epochs 1 \
      --min_clients $n \
      --run_dir results/exp6_scaling_n${n} \
      --save_name fedavg_geo_n${n}
done
```

---

## 3. Sensitive-features ablation (Phase E.1)

Re-run experiments 1, 4, 5 with `--no-drop_sensitive_features` to expose the bias-mitigation effect:

```bash
# Examples 1, 4, 5 with sensitive features kept
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Central \
    --partition_strategy iid --num_clients 1 --client_id 0 \
    --epochs 50 --no-drop_sensitive_features \
    --run_dir results/exp1_centralized_KEEP_SENSITIVE \
    --save_name centralized_keep_sensitive

python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --fl_strategy FedAvg \
    --partition_strategy geographic --num_clients 5 --rounds 100 \
    --epochs 1 --min_clients 5 --no-drop_sensitive_features \
    --run_dir results/exp4_fedavg_geo_KEEP_SENSITIVE \
    --save_name fedavg_geo_keep_sensitive
# (and the equivalent for exp5)
```

Each run records the actually-dropped column list in `partition_stats.json["dropped_sensitive_columns"]` (Phase A.8 fix) so the ablation table can cite the exact column set per run.

---

## 4. Figures (Phase E.4)

### 4.1 Per-client class distribution

```bash
python -m Analysis.CommunitiesCrime.plot_per_client_distribution \
    --run_dir results/exp4_fedavg_geo \
    --output results/figures/per_client_distribution_geo_n5.png
```

Outline §6.7 figure — bar chart of train-set class counts per client. Shows the non-IID character of the geographic partition.

### 4.2 Headline convergence figure (centralized vs federated)

```bash
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedAvg-IID=results/exp3_fedavg_iid \
           FedAvg-geo=results/exp4_fedavg_geo \
           FedProx-geo=results/exp5_fedprox_geo \
    --centralized-baseline 0.62 \
    --output results/figures/headline.png
```

The `0.62` is the macro-F1 from experiment 1's evaluation log (substitute your real number). This is the paper's primary plot per outline §7.9.

### 4.3 Scaling figure

```bash
python -m Analysis.CommunitiesCrime.plot_scaling_n_clients \
    --runs 3=results/exp6_scaling_n3 \
           5=results/exp6_scaling_n5 \
           10=results/exp6_scaling_n10 \
    --output results/figures/scaling_n_clients.png
```

### 4.4 Other plottable metrics

The headline plot script (`plot_centralized_vs_federated`) accepts any
metric the server log records via `--metric`. Useful examples:

```bash
# FedProx proximal-term evolution (Phase D + Phase E review #4) — shows the
# (μ/2)·Σ‖w-g‖² penalty's per-round average across clients.
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedProx-geo=results/exp5_fedprox_geo \
    --metric proximal_contribution \
    --output results/figures/proximal_evolution.png

# Federation-overhead — bytes uploaded per round (post-serialization).
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedAvg-N5=results/exp4_fedavg_geo \
           FedAvg-N10=results/exp6_scaling_n10 \
    --metric parameter_update_wire_bytes \
    --output results/figures/overhead_bytes.png

# Fairness — variance of per-client accuracy over rounds.
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedAvg-geo=results/exp4_fedavg_geo \
           FedProx-geo=results/exp5_fedprox_geo \
    --metric fairness_accuracy_variance \
    --output results/figures/fairness_over_rounds.png
```

The full list of metrics in each row of `server_evaluation.log`:
`aggregated_loss`, `threat_macro_f1`, `threat_accuracy`, `escalation_mae`,
`escalation_auroc`, `escalation_spearman`, `fairness_macro_f1_variance`,
`fairness_accuracy_variance`, `round_seconds`, `parameter_update_wire_bytes`,
`parameter_update_wire_bytes_per_client`, `proximal_contribution`,
`plateau_detected`, `rounds_since_improvement`. Use any of them as
`--metric`.

---

## 5. Reading the run artifacts

Every `run_dir` produced by the pipeline contains:

```
<run_dir>/
├── partitions/
│   ├── client_<i>.pkl              # train + val DataFrames per client
│   └── global_test.pkl             # frozen shared test set
├── scaler.joblib                   # fitted StandardScaler/MinMaxScaler
├── partition_stats.json            # per-client distribution, audit, dropped sensitive cols
├── env_pip_freeze.txt              # captured environment
├── <timestamp>_training.log        # per-process training log (centralized + per-client under FL)
├── <timestamp>_evaluation.log
├── server_evaluation.log           # FL only — per-round aggregated metrics from FusionFedAvg
├── client_<i>/                     # FL only — per-client log dirs
│   ├── training.log
│   └── evaluation.log
└── fed_fusion_mlp_<save_name>.keras  # FL only — final aggregated model
```

The plot scripts parse these via [Analysis/CommunitiesCrime/log_parser.py](../Analysis/CommunitiesCrime/log_parser.py) — `parse_server_log`, `parse_client_log`, `parse_partition_stats`. Use them directly for custom plots.

---

## 6. Reproducibility

* All experiments use `--commcrime_random_seed 42` by default. Pass the same value across runs to get the same global test split.
* `partition()` raises `ValueError` if you try to re-run with conflicting inputs against the same `--run_dir` (Phase A review #2). Use a fresh `--run_dir` per strategy/N variant.
* The fitted scaler is persisted in `<run_dir>/scaler.joblib` on first call; re-runs into the same dir re-load it.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: 'Config'` when running `TrainingClient.py` | CWD not project root | `cd` to project root first |
| `partition()` raises with "conflict" | Re-running with different strategy/seed into the same `--run_dir` | Use a fresh `--run_dir` |
| All clients report 0 macro-F1 / labels look degenerate | Crime-rate columns missing (mismatch with UCI schema) | Check the constants in `commCrimeDatasetLoad.py`; see §1 validation |
| FedProx run produces same macro-F1 as FedAvg | `--fedprox_mu` too small | Try `--fedprox_mu 0.1` or `--fedprox_mu 0.5` |
| `--mode hermes --model_type FUSION-MLP` → `SystemExit` | FUSION-MLP doesn't support hermes | Use `--mode legacy` (the default) |

For the full list of known limitations — particularly things never validated against real UCI data — see [implementation plan §9](Fusion_Centers_FL_Update_Implementation_Plan.md#9-known-limitations--untested-boundaries).
