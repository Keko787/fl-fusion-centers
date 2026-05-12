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

### 0.0 Activate the venv

Every command in this runbook assumes the project venv is activated:

```bash
# Linux / macOS / WSL — from the project root
source .venv/bin/activate

# Windows (PowerShell 7+)
& .\.venv\Scripts\Activate.ps1
```

The shell prompt picks up a `(.venv)` prefix when it's active. The
bootstrap scripts in §0.1 below create this venv automatically. If you
open a new terminal — including SSH'ing into a fresh host or client
node — re-run the activation command before invoking anything.

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
# Windows — works with both Windows PowerShell 5.1 (built-in) and PowerShell 7+

# Built-in PowerShell, default `python` on PATH:
powershell -ExecutionPolicy Bypass -File AppSetup\setup_fusion_node.ps1
powershell -ExecutionPolicy Bypass -File AppSetup\setup_fusion_node.ps1 -Verify

# Built-in PowerShell, using the `py` launcher (recommended — bypasses the
# Microsoft Store python.exe stub even if `python` would otherwise hit it):
powershell -ExecutionPolicy Bypass -File AppSetup\setup_fusion_node.ps1 -Verify -PythonBin py

# PowerShell 7+ (install once via `winget install Microsoft.PowerShell`):
# pwsh -File AppSetup\setup_fusion_node.ps1 -Verify -PythonBin py
```

If `pwsh` reports "term not recognized," that's PowerShell 7+ — a
separate install from the `powershell.exe` Windows ships with. The
script is compatible with both, so use the `powershell` form above. The
`-ExecutionPolicy Bypass` flag is only needed when the default execution
policy blocks unsigned local scripts (the default on a fresh Windows
install).

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

> **Single-node simulation vs. real multi-node.** Every host command in
> §2 runs `HFLHost.py` *without* `--distributed`, which means clients
> are spawned as Ray actors in the same process (the deterministic,
> reproducible default). To run the same configurations across real
> networked machines instead, add `--distributed` on the host and
> launch one `TrainingClient.py --trainingArea Federated --custom-host
> <ip>` per client machine — see §7 for the full procedure and
> network-debug checklist.

Supported `--num_clients` values: **1, 2, 3, 5, 10**. The N=2
configuration is useful for two-node hardware setups; the geographic
partitioner uses an East+Central vs. West split for N=2.

> **Which mode should I run?** Each FL experiment below lists both a
> simulation command and a real multi-node variant — they answer
> different questions. See [§2.7](#27-choosing-simulation-vs-real-multi-node-for-experiments-46)
> for per-experiment recommendations before you pick.

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

There's no host process for this experiment — `--num_clients 1` means
a single client sees the union of every training partition and trains
in one process. The command above IS the entire experiment; nothing
else runs.

Sets the macro-F1 ceiling. Phase B DoD threshold is `> 0.55` on the global test set. After this run, capture the number from `<run_dir>/<timestamp>_evaluation.log` for the `--centralized-baseline` argument to the headline plot.

**Feeds figure:** §4.2 (headline convergence — its macro-F1 is the
horizontal reference line).

### 2.2 Experiment 2 — Local-only (per client, no FL)

**One machine, sequential loop:**

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

**One machine per client (run on each machine with the right `--client_id`):**

```bash
# On each of 5 machines, vary --client_id from 0 to 4
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP \
    --trainingArea Central \
    --partition_strategy geographic \
    --num_clients 5 \
    --client_id 0 \
    --epochs 50 \
    --run_dir results/exp2_local_only \
    --save_name local_only_n5_c0
```

There's no host process for this experiment — each agency trains in
isolation on its own partition. Lower-bound result — agencies operating
in isolation. Use these client-by-client macro-F1 values to compute the
"siloed performance" baseline.

**Feeds figure:** none of the three primary plots directly — the
local-only macro-F1 numbers are reported in tables, not on the
convergence curve.

### 2.3 Experiment 3 — FedAvg / IID

**Host (single-node simulation — default):**

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

**Real multi-node variant** — add `--distributed` on the host and run
five `TrainingClient.py` processes, one per machine, varying
`--client_id` from 0 to 4:

```bash
# Host (one machine, opens [::]:8080):
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --distributed \
    --fl_strategy FedAvg --partition_strategy iid \
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 \
    --commcrime_random_seed 42 \
    --run_dir results/exp3_fedavg_iid --save_name fedavg_iid_n5

# Each client (run on its own machine, vary --client_id 0..4):
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Federated \
    --custom-host <HOST_IP> \
    --partition_strategy iid --num_clients 5 --client_id 0 \
    --commcrime_random_seed 42 --epochs 1 \
    --run_dir results/exp3_fedavg_iid --save_name fedavg_iid_n5_c0
```

Sanity check — federation works in the easy case. Phase C DoD: macro-F1 within ±0.02 of centralized.

**Feeds figure:** §4.2 (headline convergence — appears as the
`FedAvg-IID` series).

### 2.4 Experiment 4 — FedAvg / non-IID geographic

**Host (single-node simulation — default):**

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

**Real multi-node variant:**

```bash
# Host:
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --distributed \
    --fl_strategy FedAvg --partition_strategy geographic \
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 \
    --commcrime_random_seed 42 \
    --run_dir results/exp4_fedavg_geo --save_name fedavg_geo_n5

# Each client (vary --client_id 0..4 across 5 machines):
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Federated \
    --custom-host <HOST_IP> \
    --partition_strategy geographic --num_clients 5 --client_id 0 \
    --commcrime_random_seed 42 --epochs 1 \
    --run_dir results/exp4_fedavg_geo --save_name fedavg_geo_n5_c0
```

Core result — realistic fusion-center scenario.

**Feeds figures:** §4.1 (per-client class distribution — shows the
non-IID character of this run's geographic partition) and §4.2
(headline convergence — `FedAvg-geo` series).

### 2.5 Experiment 5 — FedProx / non-IID geographic

**Host (single-node simulation — default):**

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

**Real multi-node variant:**

```bash
# Host:
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --distributed \
    --fl_strategy FedProx --fedprox_mu 0.01 \
    --partition_strategy geographic \
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 \
    --commcrime_random_seed 42 \
    --run_dir results/exp5_fedprox_geo --save_name fedprox_geo_n5

# Each client (vary --client_id 0..4 across 5 machines):
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Federated \
    --custom-host <HOST_IP> \
    --partition_strategy geographic --num_clients 5 --client_id 0 \
    --commcrime_random_seed 42 --epochs 1 \
    --run_dir results/exp5_fedprox_geo --save_name fedprox_geo_n5_c0
```

The client command is identical to experiment 4 — FedProx's proximal
term `μ` lives only in the strategy on the host side and is broadcast
to clients in the per-round fit config; nothing in the client CLI
changes between FedAvg and FedProx runs.

Robustness-to-drift claim — Phase D DoD: macro-F1 ≥ FedAvg on the same partition.

**Feeds figures:** §4.2 (headline convergence — `FedProx-geo` series)
and §4.4 (proximal-term evolution via `--metric proximal_contribution`).

### 2.6 Experiment 6 — Scaling sweep

**Host (single-node simulation, sweep over N — default):**

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

**Real multi-node variant (per N)** — pick the value of N you have
hardware for, then start one host process and N client processes:

```bash
# Host (pick N ∈ {2, 3, 5, 10}; example: N=3):
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --distributed \
    --fl_strategy FedAvg --partition_strategy geographic \
    --num_clients 3 --rounds 100 --epochs 1 --min_clients 3 \
    --commcrime_random_seed 42 \
    --run_dir results/exp6_scaling_n3 --save_name fedavg_geo_n3

# Each client (run on its own machine, vary --client_id 0..N-1):
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Federated \
    --custom-host <HOST_IP> \
    --partition_strategy geographic --num_clients 3 --client_id 0 \
    --commcrime_random_seed 42 --epochs 1 \
    --run_dir results/exp6_scaling_n3 --save_name fedavg_geo_n3_c0
```

Every machine must agree on `--num_clients` and `--partition_strategy`
— see §7.3 for the full pinned-args list.

**Feeds figure:** §4.3 (scaling — best smoothed macro-F1 + rounds to
convergence vs N). Needs all three N runs to produce the bar/line
plot.

### 2.7 Choosing simulation vs. real multi-node for experiments 4–6

Every FL experiment in §2 documents both a single-node simulation
command and a real multi-node variant. They're not interchangeable —
pick based on what claim the experiment is meant to support.

**Trade-offs:**

| Dimension | Single-node simulation | Real multi-node |
|---|---|---|
| Determinism | Bit-reproducible at a given seed | Non-deterministic (network jitter, OS scheduling, TF drift) |
| Wall-clock per run | Fast (no gRPC, one TF startup) | Slower (gRPC + per-machine TF warmup × N) |
| Hardware needed | 1 machine | N+1 machines + working network |
| Faithful to deployment? | No — skips the gRPC stack, firewall, real I/O | Yes — exercises the whole stack |
| Scaler default | Global (union of all clients' training rows) | Per-client (each client fits locally — §7.3.1) |
| Sim ↔ multi-node match | — | Pass `--global_scaler` on every client to match sim bit-for-bit |
| Debuggability | One stack trace, one log dir | Logs scattered across N+1 machines |
| Supports a claim like… | "FedProx beats FedAvg under non-IID drift" | "This works on Chameleon / real hardware" |

**Per-experiment recommendation:**

| Experiment | Primary mode | Multi-node use case |
|---|---|---|
| **Exp 4** — FedAvg / non-IID geographic (headline accuracy) | **Simulation.** The paper's headline number — needs bit-reproducibility. | One multi-node run with `--global_scaler` as a deployment-validation supplement. |
| **Exp 5** — FedProx / non-IID geographic (robustness) | **Simulation.** Cleanest A/B against FedAvg on the same partition; network nondeterminism would only add noise to the proximal-term comparison. | Only if you're making a separate stragglers / failure-recovery claim about FedProx — otherwise skip. |
| **Exp 6** — Scaling sweep (N=3, 5, 10) | **Simulation only.** Scaling is an algorithmic claim (does macro-F1 degrade as N grows?). Real multi-node would conflate algorithmic and systems scaling, and standing up 10 real machines for one bar chart is overkill. | Run a separate "systems scaling" figure with `--metric round_seconds` and `--metric parameter_update_wire_bytes` if you want wall-clock and wire-overhead numbers — that's a different plot, not the macro-F1 scaling claim. |

**A clean publishable workflow:**

1. Run all six experiments in §2 in simulation with a fixed
   `--commcrime_random_seed`. These produce the reproducible headline
   numbers in the paper.
2. Re-run **one** experiment (typically Exp 4, since it's the core
   non-IID result) in real multi-node mode with `--global_scaler` and
   the same seed. Cite it as deployment validation: "the simulation
   result is confirmed by an end-to-end multi-node run with
   macro-F1 = X.XX."
3. Save Exp 5 / Exp 6 multi-node runs for follow-on work — only run
   them now if a reviewer specifically asks or your deployment story
   needs it.

**When to deviate:** if your hardware allocation is already N nodes
(e.g., a long-running Chameleon reservation), real multi-node costs
roughly the same effort as simulation. In that case run every
experiment both ways: simulation for the paper, multi-node for your
own confidence, cite the sim numbers in the writeup.

**One nuance about the realism vs. fidelity trade-off:** the per-client
scaler default for multi-node is actually *more* realistic than the
simulation's global scaler (real fusion centers can't see each other's
data even for normalization). `--global_scaler` is a convenience for
direct sim-vs-multi-node comparison, not the privacy-preserving
default. Pick consciously; document which mode each headline number
came from.

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

All plot scripts live in [`Analysis/CommunitiesCrime/`](../Analysis/CommunitiesCrime/)
and read directly from the run artifacts in §5 — no separate logging
pass needed, just point each script at the relevant `--run_dir`(s).

**Activate the venv before running any plot command** (the scripts
import matplotlib, joblib, and the project's own log parser):

```bash
source .venv/bin/activate   # Linux / macOS / WSL
```

The three primary scripts and what they consume:

| Script | Reads from | Produces |
|---|---|---|
| `plot_per_client_distribution` | one `--run_dir` (uses its `partition_stats.json`) | bar chart of train-set class counts per client |
| `plot_centralized_vs_federated` | one or more `--run_dir`s (uses their `server_evaluation.log`) | overlaid metric-vs-round line plot, with optional horizontal baseline |
| `plot_scaling_n_clients` | N `--run_dir`s tagged by client count | bar of best macro-F1 + line of rounds-to-convergence vs N |

All three accept:

- `--output <path>` — extension (`.png`, `.pdf`, `.svg`, `.eps`)
  picks the format.
- `--style {default,paper}` — `paper` uses 300 DPI, no top/right
  spines, and larger fonts; use this for camera-ready figures.

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

## 7. Real multi-node federation

Single-process simulation (§2) is great for reproducibility but doesn't
prove anything about wire-time behavior, firewall rules, or partition
determinism across machines. To run any of the §2 experiments across
**real, networked nodes**, add `--distributed` to the host invocation
and launch each client as its own process on its own machine.

### 7.1 Topology

```
┌──────────────────┐               ┌──────────────────┐
│  Host machine    │               │  Client machine  │
│ HFLHost.py       │◄── gRPC 8080 ─┤ TrainingClient.py│
│ --distributed    │               │ --trainingArea   │
│ binds [::]:8080  │               │   Federated      │
└──────────────────┘               │ --custom-host …  │
        ▲                          └──────────────────┘
        │  also gRPC 8080
        │
   one client process per machine, one per --client_id
```

The host process must outlive every client. The Flower server stays at
`[ROUND 1]` (prints that label *before* clients are required) until
`--min_clients` clients have connected; round 1 only fires after the
roster fills.

### 7.2 Commands

**On the host machine** (activate venv first):

```bash
source .venv/bin/activate
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --distributed --fl_strategy FedAvg \
    --partition_strategy iid --num_clients 3 \
    --rounds 10 --min_clients 3 --epochs 1 \
    --commcrime_random_seed 42 \
    --run_dir results/fed_run1 --save_name fed_run1
```

Wait for the line `=== FUSION-MLP distributed server on [::]:8080 ===`
before launching clients.

**On each client machine** (one invocation each, varying `--client_id`):

```bash
source .venv/bin/activate
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Federated \
    --custom-host <HOST_IP> \
    --partition_strategy iid --num_clients 3 --client_id 0 \
    --commcrime_random_seed 42 --epochs 1 \
    --run_dir results/fed_run1 --save_name client0
```

`<HOST_IP>` is whatever IP the host machine is reachable on from this
client (LAN, floating IP, or VPN address — must match the side that
got opened in §7.4). Repeat with `--client_id 1`, `2`, etc. on the
other client machines.

### 7.3 Partition determinism across machines

Every client AND the host must pass identical values for:

- `--num_clients`
- `--partition_strategy` (+ `--dirichlet_alpha` if dirichlet)
- `--commcrime_random_seed`
- `--drop_sensitive_features` / `--no-drop_sensitive_features`
- `--global_scaler` (see §7.3.1) — must be set the same way on every
  client, or different clients will train against features on
  incompatible scales

If any of these differ, each client deterministically computes its own
(different) partition or features, and FedAvg silently averages
weights trained against incompatible distributions. Macro-F1 numbers
will look weird but won't error.

> The two-client (`--num_clients 2`) configuration is now supported.
> Use it for two-node testbed setups; partitioning works for iid /
> geographic / dirichlet strategies. Note that geographic splits are
> imbalanced at N=2 (~3.7× more training rows in East+Central than in
> West) by design.

#### 7.3.1 Global vs per-client scaler — and how to match the simulation

Two preprocessing modes are available:

| Mode | Flag | Scaler fit on | Use when |
|---|---|---|---|
| **Per-client (default)** | _(omit)_ | each client's local training rows only | the realistic privacy story — clients never see each other's data, even at preprocessing time |
| **Global** | `--global_scaler` | union of every client's training rows | reproducing the simulation runner's results bit-for-bit at the same `--commcrime_random_seed` |

The simulation runner uses the global scaler unconditionally (outline
§6.5 "global standardization"). Distributed clients default to
per-client scalers because in a true fusion-center deployment each
agency cannot see another's data. The cost: per-client and global
scalers can disagree by **10–16 standard deviations** on some
features (verified on the iid / N=3 / seed=42 split), which causes
the multi-node macro-F1 to drift from the simulation's even when the
seed matches.

To make multi-node and simulation results directly comparable, pass
`--global_scaler` on every client:

```bash
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Federated \
    --custom-host <HOST_IP> \
    --partition_strategy iid --num_clients 3 --client_id 0 \
    --commcrime_random_seed 42 --epochs 1 \
    --global_scaler \
    --run_dir results/fed_run1 --save_name client0_gs
```

This requires every distributed client to have the full COMMCRIME raw
archive on its local filesystem. Since partitioning is deterministic
from `--commcrime_random_seed`, every node independently computes the
same global scaler — no out-of-band file shipping needed. The cost is
a small amount of redundant compute per client (~2 KB of CSV, fits the
scaler in well under a second).

If you want the realistic privacy threat model (per-client scalers),
omit the flag and accept a small fidelity gap from the simulation.

### 7.4 Network requirements & first-time setup

- **Port 8080 inbound** must be open on the host machine. The Flower
  server is plaintext gRPC (no TLS) — same threat model as the
  simulation runs. If you're on Chameleon, ensure the security group
  allows TCP/8080 from the client subnet. If you're on a Linux host
  with `ufw` active, run `sudo ufw allow 8080/tcp` once.
- **The port is hardcoded to 8080** in `TrainingClient.py`. Don't try
  to bind the server to a different port without also editing the
  client wiring.
- **Verify TCP reachability before launching the FL processes.** From
  any client machine:
  ```bash
  nc -vz <HOST_IP> 8080
  ```
  The three useful outcomes:

  | `nc` says | Meaning | Fix |
  |---|---|---|
  | `Connection ... succeeded` | host process is bound and the firewall is open | proceed |
  | `Connection refused` | host kernel responded with RST — port is open but **no server process is listening** | start (or restart) the host with `--distributed` |
  | `No route to host` / `connection timed out` | firewall (host or cloud SG) is dropping the SYN | open TCP/8080; ICMP working (`ping`) doesn't mean TCP works |

- **ICMP ≠ TCP.** A successful `ping` only proves the kernel routes
  packets to the host. Firewalls routinely allow ICMP while blocking
  application ports. Always probe with `nc` before debugging Flower.

### 7.5 What you should see

- Host: `[ROUND 1]` appears immediately, then nothing further until
  all clients connect. Once round 1 completes you'll see per-round
  metrics in `<run_dir>/server_evaluation.log` (one block per round).
- Each client: prints the dataset summary, then sits silent on the
  `start_client()` deprecation warning while the gRPC stream is open.
  Real training output is per-process Keras epoch lines if `--epochs`
  is > 1, otherwise just the round-boundary entries in the client's
  `<run_dir>/<timestamp>_training.log`.
- Round-1 latency includes every client's first-epoch warmup (TF graph
  compile + scaler fit). Subsequent rounds finish in well under a
  second per round for FUSION-MLP / COMMCRIME.

### 7.6 Multi-node troubleshooting checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Client hangs forever after `DEPRECATED FEATURE: start_client()` | gRPC silent connection-retry loop — TCP:8080 unreachable | Run `nc -vz <HOST_IP> 8080` from the client. `No route to host` → firewall. `Connection refused` → host not running. |
| Client raises `StatusCode.UNAVAILABLE / No route to host` | Firewall blocking inbound TCP 8080 on the host | Open the port (ufw / iptables / cloud SG) |
| All clients connect but server stays at `[ROUND 1]` | Fewer than `--min_clients` connections established | Check `sudo ss -tnp \| grep 8080` on the host — look for `ESTAB` lines, one per connected client. If they're missing despite the client logs saying "connected," it's gRPC retrying silently — confirm with `nc -vz` |
| Each client trains fine but the aggregated model is garbage | Partition-determinism arguments differ across nodes (§7.3) | Pin `--num_clients`, `--partition_strategy`, `--commcrime_random_seed`, and `--drop_sensitive_features` to the same values everywhere |
| Server reports `min_available_clients` mismatch warnings | One client crashed mid-round — Flower waited it out | Restart the dead client; the server resumes the next round when it reconnects |
| `--num_clients 2` rejected by argparse | Older copy of `ArgumentConfigLoad.py` before N=2 was added | Pull latest; the supported set is now `{1, 2, 3, 5, 10}` |

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: 'Config'` when running `TrainingClient.py` | CWD not project root | `cd` to project root first |
| `partition()` raises with "conflict" | Re-running with different strategy/seed into the same `--run_dir` | Use a fresh `--run_dir` |
| All clients report 0 macro-F1 / labels look degenerate | Crime-rate columns missing (mismatch with UCI schema) | Check the constants in `commCrimeDatasetLoad.py`; see §1 validation |
| FedProx run produces same macro-F1 as FedAvg | `--fedprox_mu` too small | Try `--fedprox_mu 0.1` or `--fedprox_mu 0.5` |
| `--mode hermes --model_type FUSION-MLP` → `SystemExit` | FUSION-MLP doesn't support hermes | Use `--mode legacy` (the default) |

For multi-node / real-network issues, see §7.6 instead. For the full
list of known limitations — particularly things never validated against
real UCI data — see [implementation plan §9](Fusion_Centers_FL_Update_Implementation_Plan.md#9-known-limitations--untested-boundaries).
