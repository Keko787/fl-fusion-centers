# HFL-DNN-GAN-NIDS / Fusion Centers FL

A federated-learning research codebase with two complementary stacks:

1. **HFL-DNN-GAN-NIDS** *(original)* — Hierarchical Federated Learning and
   GAN-based Network Intrusion Detection for private and IoT networks
   (CIC IoT2023 / IoTBotnet2020). Drone-fleet deployment via AERPAW.
2. **Fusion Centers FL Update** *(2026)* — flat federated multi-task MLP
   for cross-jurisdictional crime / threat-intelligence prediction on the
   UCI Communities and Crime (Unnormalized) dataset. Builds on the same
   Flower/TF infrastructure but targets a different threat model (regional
   law-enforcement fusion centers, not IoT intrusion detection).

Both stacks share the entry-point CLIs (`HFLHost.py`, `TrainingClient.py`)
and the `Config/` configuration tree. The model and dataset code paths
diverge below that.

## Table of Contents

- [Team](#team)
- [Prerequisites](#prerequisites)
- [Quickstart: Fusion Centers FL](#quickstart-fusion-centers-fl)
- [Quickstart: HFL-DNN-GAN-NIDS](#quickstart-hfl-dnn-gan-nids)
- [Datasets](#datasets)
- [Usage](#usage)
  - [Activating the venv](#activating-the-venv)
  - [Fusion Centers — Centralized Training](#fusion-centers--centralized-training)
  - [Fusion Centers — Federated Training (Single-Node Simulation)](#fusion-centers--federated-training-single-node-simulation)
  - [Fusion Centers — Federated Training (Real Multi-Node)](#fusion-centers--federated-training-real-multi-node)
  - [Fusion Centers — Generating Figures](#fusion-centers--generating-figures)
  - [NIDS — Federated Training (Host)](#nids--federated-training-host)
  - [NIDS — Localized & Federated Training (Client)](#nids--localized--federated-training-client)
- [Repository Layout](#repository-layout)
- [Architecture](#architecture)
- [Models](#models)
- [Documentation](#documentation)
- [Testing](#testing)
- [License](#license)

---

## Team

**Faculty Advisors (2026)**

- Dr. Chenqi Qu
- Dr. Prasad Calyam
- Dr. Alicia Morel

**Graduate Students (2026)**
- Kevin Kostage

---

## Prerequisites

| Stack | OS | Python | Hardware |
|---|---|---|---|
| Fusion Centers FL | Ubuntu 22.04 LTS | 3.9+ | CPU is fine (outline §7.3 — model is small) |
| HFL-DNN-GAN-NIDS | Ubuntu 22.04 LTS | 3.8+ | CUDA 12 drivers (P100 / M40 supported) |

Python 3.9 is the minimum for Fusion Centers because
`argparse.BooleanOptionalAction` is used by the `--drop_sensitive_features`
flag. TensorFlow 2.15–2.21 and Flower 1.19+ are supported.

**On Windows:** make sure `python --version` prints an actual version
(`3.x.y`) before running the bootstrap script. If it prints "Python was
not found, run without arguments to install from the Microsoft Store…",
that's the Windows app-execution-alias stub — real Python isn't
installed. Install it with:

```powershell
winget install --id Python.Python.3.12 --source winget
```

…then **close and reopen the terminal** so the new PATH is picked up.

If `python` still resolves to the stub even after install (you can
check with `where.exe python` — the stub at `WindowsApps\python.exe`
will appear first), you have three options:

1. **Use the `py` launcher** — install adds `C:\Windows\py.exe` which
   is immune to the alias issue. Pass it to the setup script:
   ```powershell
   powershell -ExecutionPolicy Bypass -File AppSetup\setup_fusion_node.ps1 -Verify -PythonBin py
   ```
   The script also auto-falls-back to `py` if it detects the stub, so
   `-PythonBin py` is only needed for older versions of the script.
2. **Disable the Microsoft Store alias** in
   **Settings → Apps → Advanced app settings → App execution aliases**
   (toggle off `python.exe` and `python3.exe`), then reopen the
   terminal.
3. **Reorder PATH** so `C:\Users\<you>\AppData\Local\Programs\Python\PythonNNN\`
   appears before `WindowsApps`.

---

## Quickstart: Fusion Centers FL

The Fusion Centers stack ships with a one-shot bootstrap that creates a
venv, installs pinned deps, downloads the UCI Communities and Crime
archive into `$HOME/datasets/CommunitiesCrime/`, generates the `.names`
schema, and validates the install.

```bash
# Clone
git clone https://github.com/Keko787/fl-fusion-centers.git
cd fl-fusion-centers

# Linux / macOS / WSL — full setup + run the test suite to verify
bash AppSetup/setup_fusion_node.sh --verify

# Activate the venv it created
source .venv/bin/activate
```

```powershell
# Windows — works with both Windows PowerShell 5.1 (built-in) and PowerShell 7+
git clone https://github.com/Keko787/fl-fusion-centers.git
cd fl-fusion-centers

# Option A: use the PowerShell that ships with Windows (no install needed)
powershell -ExecutionPolicy Bypass -File AppSetup\setup_fusion_node.ps1 -Verify

# Option A' (recommended on Windows): pass -PythonBin py to use the py launcher.
#   Bypasses the Microsoft Store python.exe stub entirely. Safe to use even when
#   `python --version` works correctly — `py.exe` is the standard launcher
#   shipped with the python.org / winget Python install.
powershell -ExecutionPolicy Bypass -File AppSetup\setup_fusion_node.ps1 -Verify -PythonBin py

# Option B: use PowerShell 7+ (install once via winget)
#   winget install --id Microsoft.PowerShell --source winget
#   then in a new pwsh terminal:
# pwsh -File AppSetup\setup_fusion_node.ps1 -Verify -PythonBin py

# Activate the venv it created (any PowerShell):
& .\.venv\Scripts\Activate.ps1
```

> **`pwsh` not found?** That command is PowerShell 7+, which is a
> separate install from the `powershell.exe` that ships with Windows.
> Use Option A above (the script is compatible with both versions —
> 5.1's built-in `powershell` works identically). The
> `-ExecutionPolicy Bypass` flag is only needed if your machine's
> default execution policy blocks unsigned local scripts; you can
> drop it if you've run `Set-ExecutionPolicy -Scope CurrentUser
> RemoteSigned` previously.

Container alternative:

```bash
# Linux / macOS / WSL
docker build -t fusion-centers -f AppSetup/DockerSetup/Dockerfile.fusion .
docker run --rm -v "$PWD/results:/app/results" fusion-centers \
    python App/TrainingApp/Client/TrainingClient.py --model_type FUSION-MLP --help
```

```powershell
# Windows (PowerShell — built-in 5.1 or pwsh 7+; requires Docker Desktop)
docker build -t fusion-centers -f AppSetup\DockerSetup\Dockerfile.fusion .
docker run --rm -v "${PWD}\results:/app/results" fusion-centers `
    python App/TrainingApp/Client/TrainingClient.py --model_type FUSION-MLP --help
```

Then follow [`DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md`](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md)
for the six-experiment matrix that reproduces the paper figures.

---

## Quickstart: HFL-DNN-GAN-NIDS

> **Windows note:** the NIDS provisioning scripts target Linux testbeds
> (AERPAW drones, Chameleon bare-metal). They will not run cleanly on
> native Windows — they assume `apt`, Linux paths, and POSIX shell
> primitives. Use WSL2 or run them on the target Linux host directly.

```bash
# Linux / macOS / WSL — clone the repository
git clone https://github.com/Keko787/HFL-DNN-GAN-IDS.git
cd HFL-DNN-GAN-IDS

# [Option 1] AERPAW node setup (drone-fleet testbed)
python3 AppSetup/AERPAW_node_Setup.py

# [Option 2] Chameleon node setup (CPU bare-metal)
python3 AppSetup/Chameleon_node_Setup.py
```

```powershell
# Windows (PowerShell) — clone only; provisioning must run inside WSL2 or on a Linux node
git clone https://github.com/Keko787/HFL-DNN-GAN-IDS.git
cd HFL-DNN-GAN-IDS

# Then drop into WSL to run the Linux-only provisioning:
#   wsl bash -c "python3 AppSetup/AERPAW_node_Setup.py"
#   wsl bash -c "python3 AppSetup/Chameleon_node_Setup.py"
```

The NIDS stack uses `requirements_core.txt` (TF 2.15, Flower 1.9,
tensorflow-privacy 0.9.0). Don't mix it with `requirements_fusion.txt`
in the same venv — they pin incompatible TF versions.

---

## Datasets

### Fusion Centers FL — UCI Communities and Crime (Unnormalized)

Auto-fetched by `AppSetup/setup_fusion_node.{sh,ps1}`. Manual install:

```bash
# Linux / macOS / WSL
mkdir -p ~/datasets/CommunitiesCrime
curl -L https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip \
    -o /tmp/comm_crime.zip
unzip -j /tmp/comm_crime.zip CommViolPredUnnormalizedData.txt -d ~/datasets/CommunitiesCrime/

# Regenerate the .names schema (UCI 2.0 dropped it from the zip)
pip install ucimlrepo
python -m Config.DatasetConfig.CommunitiesCrime_Sampling.generate_names_file
```

```powershell
# Windows (PowerShell)
$dest = "$env:USERPROFILE\datasets\CommunitiesCrime"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip" `
    -OutFile "$env:TEMP\comm_crime.zip"
Expand-Archive -Path "$env:TEMP\comm_crime.zip" -DestinationPath "$env:TEMP\comm_crime" -Force
Copy-Item "$env:TEMP\comm_crime\CommViolPredUnnormalizedData.txt" $dest -Force

# Regenerate the .names schema (UCI 2.0 dropped it from the zip)
pip install ucimlrepo
python -m Config.DatasetConfig.CommunitiesCrime_Sampling.generate_names_file
```

After install, validate the schema with the 5-line snippet in
[`RUNNING_FUSION_EXPERIMENTS.md` §1](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md#1-data-acquisition).

### NIDS — CIC IoT2023 / IoTBotnet2020

1. Download **CIC IoT2023** from [CIC website](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
2. Upload `CICIoT2023.zip` to `$HOME/datasets/`, then:

```bash
# Linux / macOS / WSL
unzip $HOME/datasets/CICIoT2023.zip -d $HOME/datasets/CICIOT2023
```

```powershell
# Windows (PowerShell)
Expand-Archive -Path "$env:USERPROFILE\datasets\CICIoT2023.zip" `
    -DestinationPath "$env:USERPROFILE\datasets\CICIOT2023" -Force
```

---

## Usage

All commands assume the project root is the current directory and the
appropriate venv is activated.

### Activating the venv

Before running any command in this section, activate the venv that
`AppSetup/setup_fusion_node.sh` (or your manual install) created:

```bash
# Linux / macOS / WSL — from the project root
source .venv/bin/activate
```

```powershell
# Windows (PowerShell 7+)
& .\.venv\Scripts\Activate.ps1
```

Your shell prompt should now show a `(.venv)` prefix. Run `deactivate`
to leave the venv. If you Ctrl-C a training run or open a new terminal,
remember to re-activate.

### Fusion Centers — Centralized Training

Single-process baseline. Sets the macro-F1 ceiling federated runs are
compared against.

```bash
# Linux / macOS / WSL
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Central \
    --partition_strategy iid --num_clients 1 --client_id 0 \
    --epochs 50 \
    --run_dir results/exp1_centralized \
    --save_name centralized_baseline
```

```powershell
# Windows (PowerShell)
python App/TrainingApp/Client/TrainingClient.py `
    --model_type FUSION-MLP --trainingArea Central `
    --partition_strategy iid --num_clients 1 --client_id 0 `
    --epochs 50 `
    --run_dir results/exp1_centralized `
    --save_name centralized_baseline
```

### Fusion Centers — Federated Training (Single-Node Simulation)

Default mode — every "client" is a Ray actor in the same process via
`fl.simulation.start_simulation`. Use this for reproducible local
experiments. Five-client FedAvg over a geographic non-IID partition
(the realistic fusion-center scenario):

```bash
# Linux / macOS / WSL
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --fl_strategy FedAvg \
    --partition_strategy geographic \
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 \
    --run_dir results/exp4_fedavg_geo \
    --save_name fedavg_geo_n5
```

```powershell
# Windows (PowerShell)
python App/TrainingApp/HFLHost/HFLHost.py `
    --model_type FUSION-MLP --fl_strategy FedAvg `
    --partition_strategy geographic `
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 `
    --run_dir results/exp4_fedavg_geo `
    --save_name fedavg_geo_n5
```

FedProx variant (drift-robust on non-IID partitions):

```bash
# Linux / macOS / WSL
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --fl_strategy FedProx --fedprox_mu 0.01 \
    --partition_strategy geographic \
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 \
    --run_dir results/exp5_fedprox_geo \
    --save_name fedprox_geo_n5
```

```powershell
# Windows (PowerShell)
python App/TrainingApp/HFLHost/HFLHost.py `
    --model_type FUSION-MLP --fl_strategy FedProx --fedprox_mu 0.01 `
    --partition_strategy geographic `
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 `
    --run_dir results/exp5_fedprox_geo `
    --save_name fedprox_geo_n5
```

Supported `--num_clients` values: **1, 2, 3, 5, 10** (geographic
partitioner ships matching bucket tables for each).

Full experiment matrix (including local-only baseline, IID FedAvg, and
the N ∈ {3, 5, 10} scaling sweep) is in
[`RUNNING_FUSION_EXPERIMENTS.md`](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md).

### Fusion Centers — Federated Training (Real Multi-Node)

To run FL across real, networked machines instead of a single-process
simulator, add `--distributed` to the host invocation. The host binds a
Flower gRPC server on `[::]:8080`; each client is a separate
`TrainingClient.py --trainingArea Federated` process on its own machine
that dials the host's IP.

**Host machine** (one — must outlive the run, port 8080 open inbound):

```bash
# Linux / macOS / WSL
source .venv/bin/activate
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --distributed --fl_strategy FedAvg \
    --partition_strategy iid --num_clients 3 \
    --rounds 10 --min_clients 3 --epochs 1 \
    --commcrime_random_seed 42 \
    --run_dir results/fed_run1 --save_name fed_run1
```

```powershell
# Windows (PowerShell)
& .\.venv\Scripts\Activate.ps1
python App/TrainingApp/HFLHost/HFLHost.py `
    --model_type FUSION-MLP --distributed --fl_strategy FedAvg `
    --partition_strategy iid --num_clients 3 `
    --rounds 10 --min_clients 3 --epochs 1 `
    --commcrime_random_seed 42 `
    --run_dir results/fed_run1 --save_name fed_run1
```

**Each client machine** (one invocation per client, varying `--client_id`):

```bash
# Linux / macOS / WSL
source .venv/bin/activate
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Federated \
    --custom-host <HOST_IP> \
    --partition_strategy iid --num_clients 3 --client_id 0 \
    --commcrime_random_seed 42 --epochs 1 \
    --run_dir results/fed_run1 --save_name client0
```

```powershell
# Windows (PowerShell)
& .\.venv\Scripts\Activate.ps1
python App/TrainingApp/Client/TrainingClient.py `
    --model_type FUSION-MLP --trainingArea Federated `
    --custom-host <HOST_IP> `
    --partition_strategy iid --num_clients 3 --client_id 0 `
    --commcrime_random_seed 42 --epochs 1 `
    --run_dir results/fed_run1 --save_name client0
```

Replace `<HOST_IP>` with the host machine's reachable IP. Repeat the
client command on the other machines varying `--client_id 1`, `2`, etc.
The server blocks at `[ROUND 1]` until `--min_clients` have connected,
then rounds proceed automatically.

**Hard requirements for partitions to line up across machines** — every
client AND the host must pass identical:
- `--num_clients`, `--partition_strategy` (+ `--dirichlet_alpha` if dirichlet),
- `--commcrime_random_seed`,
- `--drop_sensitive_features` / `--no-drop_sensitive_features`.

If any of these differ, clients see different partitions and FL
silently produces garbage.

**Network requirements:**
- Port **8080** open on the host firewall (gRPC, plaintext — no TLS).
- Clients dial `<HOST_IP>:8080` — the port is currently hardcoded.
- `ping` works ≠ TCP:8080 works. Verify *before* you start the FL processes:
  - Linux / macOS / WSL: `nc -vz <HOST_IP> 8080`
  - Windows (PowerShell): `Test-NetConnection -ComputerName <HOST_IP> -Port 8080`
    (`TcpTestSucceeded : True` means the port is reachable).

**Client-count limits:** Same as simulation —
`--num_clients ∈ {1, 2, 3, 5, 10}`. For only 2 clients pass
`--num_clients 2 --min_clients 2`; the geographic partitioner uses a
2-way East+Central vs. West split.

**Bit-comparable to the simulation** — add `--global_scaler` on every
client to fit the feature scaler on the union of all clients' training
partitions (what the in-process simulator does), instead of fitting
each client's scaler locally. Without this flag, multi-node and
simulation feature matrices differ by ~10–16 standard deviations on
some features, which makes accuracy numbers diverge slightly. Every
client must have access to the full COMMCRIME raw archive locally
(partition is deterministic from the seed, so each node computes the
same global scaler). Use this flag when you want sim and multi-node
runs to produce the same macro-F1 at the same seed.

See [`RUNNING_FUSION_EXPERIMENTS.md` §7](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md#7-real-multi-node-federation)
for the firewall / port troubleshooting checklist.

### Fusion Centers — Generating Figures

After running any experiment, the three plot scripts in
[`Analysis/CommunitiesCrime/`](Analysis/CommunitiesCrime/) read directly
from the `--run_dir` artifacts (no separate logging pass needed). Make
sure the venv is active first.

```bash
# Linux / macOS / WSL
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
& .\.venv\Scripts\Activate.ps1
```

**Per-client class distribution** (uses one run's `partition_stats.json`
— shows the non-IID character of the partition):

```bash
# Linux / macOS / WSL
python -m Analysis.CommunitiesCrime.plot_per_client_distribution \
    --run_dir results/exp4_fedavg_geo \
    --output results/figures/per_client_distribution_geo_n5.png
```

```powershell
# Windows (PowerShell)
python -m Analysis.CommunitiesCrime.plot_per_client_distribution `
    --run_dir results/exp4_fedavg_geo `
    --output results/figures/per_client_distribution_geo_n5.png
```

**Headline convergence** (overlays one or more FL runs' macro-F1 over
rounds, with the centralized result as a horizontal reference line —
substitute the actual centralized macro-F1 from experiment 1's
`<run_dir>/<timestamp>_evaluation.log`):

```bash
# Linux / macOS / WSL
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated \
    --runs FedAvg-IID=results/exp3_fedavg_iid \
           FedAvg-geo=results/exp4_fedavg_geo \
           FedProx-geo=results/exp5_fedprox_geo \
    --centralized-baseline 0.62 \
    --output results/figures/headline.png
```

```powershell
# Windows (PowerShell)
python -m Analysis.CommunitiesCrime.plot_centralized_vs_federated `
    --runs FedAvg-IID=results/exp3_fedavg_iid `
           FedAvg-geo=results/exp4_fedavg_geo `
           FedProx-geo=results/exp5_fedprox_geo `
    --centralized-baseline 0.62 `
    --output results/figures/headline.png
```

**Scaling sweep** (best smoothed macro-F1 + rounds-to-convergence vs N
— needs the three runs from experiment 6):

```bash
# Linux / macOS / WSL
python -m Analysis.CommunitiesCrime.plot_scaling_n_clients \
    --runs 3=results/exp6_scaling_n3 \
           5=results/exp6_scaling_n5 \
           10=results/exp6_scaling_n10 \
    --output results/figures/scaling_n_clients.png
```

```powershell
# Windows (PowerShell)
python -m Analysis.CommunitiesCrime.plot_scaling_n_clients `
    --runs 3=results/exp6_scaling_n3 `
           5=results/exp6_scaling_n5 `
           10=results/exp6_scaling_n10 `
    --output results/figures/scaling_n_clients.png
```

All three scripts accept `--style paper` for camera-ready output (300
DPI, larger fonts, no top/right spines). The output extension picks the
format — `.png`, `.pdf`, `.svg`, `.eps` all work.

The headline script also accepts arbitrary metrics from the server log
via `--metric` (e.g. `proximal_contribution`, `parameter_update_wire_bytes`,
`fairness_accuracy_variance`). See
[`RUNNING_FUSION_EXPERIMENTS.md` §4](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md#4-figures-phase-e4)
for the full metric list and ablation/overhead figure recipes.

### NIDS — Federated Training (Host)

```bash
# Linux / macOS / WSL
python3 App/TrainingApp/HFLHost/HFLHost.py --help
```

```powershell
# Windows (PowerShell) — use `python` (the NIDS scripts assume Linux at runtime; --help works anywhere)
python App/TrainingApp/HFLHost/HFLHost.py --help
```

### NIDS — Localized & Federated Training (Client)

```bash
# Linux / macOS / WSL
python3 App/TrainingApp/Client/TrainingClient.py --help
# Default uses CICIOT2023 dataset; use --dataset IOTBOTNET for IoTBotnet.
```

```powershell
# Windows (PowerShell)
python App/TrainingApp/Client/TrainingClient.py --help
# Default uses CICIOT2023 dataset; use --dataset IOTBOTNET for IoTBotnet.
```

Demo of localized NIDS training:

```bash
# Linux / macOS / WSL
python3 App/TrainingApp/Client/TrainingClient.py \
    --model_type AC-GAN --model_training Both --trainingArea Central \
    --dataset CICIOT --save_name Test1
```

```powershell
# Windows (PowerShell)
python App/TrainingApp/Client/TrainingClient.py `
    --model_type AC-GAN --model_training Both --trainingArea Central `
    --dataset CICIOT --save_name Test1
```

---

## Repository Layout

```
fl-fusion-centers/
├── App/TrainingApp/
│   ├── Client/TrainingClient.py        # Central + per-client FL entry point
│   └── HFLHost/HFLHost.py              # FL orchestration entry point
├── Config/
│   ├── SessionConfig/                  # Args, hyperparams, model factory, run dir
│   ├── DatasetConfig/
│   │   ├── CommunitiesCrime_Sampling/  # Fusion Centers data path (Phase A)
│   │   └── …                           # Legacy NIDS dataset configs
│   ├── modelStructures/
│   │   ├── FusionMLP/                  # Multi-task MLP + FedProx variant (Phase B)
│   │   └── …                           # NIDS model structures
│   └── ModelTrainingConfig/
│       ├── ClientModelTrainingConfig/  # Central + Flower client wrappers
│       └── HostModelTrainingConfig/
│           └── FusionCenters/          # FedAvg/FedProx strategies + SimulationRunner
├── Analysis/CommunitiesCrime/          # Log parsers + plot scripts (Phase E)
├── AppSetup/
│   ├── requirements_fusion.txt         # Fusion-centers deps (minimal, TF 2.15–2.21)
│   ├── requirements_core.txt           # Legacy NIDS deps (TF 2.15, pinned)
│   ├── requirements_edge.txt           # Legacy NIDS edge/drone deps
│   ├── setup_fusion_node.sh            # One-shot Linux/macOS bootstrap
│   ├── setup_fusion_node.ps1           # One-shot Windows bootstrap
│   ├── AERPAW_node_Setup.py            # Legacy NIDS drone-fleet provisioning
│   ├── Chameleon_node_Setup.py         # Legacy NIDS Chameleon provisioning
│   └── DockerSetup/
│       ├── Dockerfile.fusion           # Fusion Centers slim image
│       ├── Dockerfile.{client,server}  # Legacy NIDS images
│       └── docker-compose.yml
├── DeveloperDocs/
│   ├── Fusion_Centers_FL_Update_Design.md
│   ├── Fusion_Centers_FL_Update_Implementation_Plan.md
│   └── RUNNING_FUSION_EXPERIMENTS.md
└── tests/                              # 183 fusion-centers tests + legacy NIDS tests
```

---

## Architecture

### Fusion Centers FL (flat federated multi-task MLP)

```
                ┌──────────────────┐
                │ Flower simulation │
                │   (FedAvg /       │
                │    FedProx)       │
                └────────┬─────────┘
        configure_fit /  │  aggregate_fit / evaluate
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ Client 0  │  │ Client 1  │  │ Client N  │
    │ FUSION-MLP│  │ FUSION-MLP│  │ FUSION-MLP│
    │ (region A)│  │ (region B)│  │ (region X)│
    └───────────┘  └───────────┘  └───────────┘

Each client trains a shared-trunk MLP with two heads:
  threat_class (softmax, 3 classes)    +    escalation_score (sigmoid)
Joint loss: α·CrossEntropy + β·BCE
```

Partitioning is geographic (FIPS → postal → region bucketing) for the
non-IID realistic case, IID for the easy baseline, and Dirichlet for the
controlled-skew ablation.

### HFL-DNN-GAN-NIDS (three-tier hierarchical)

1. **Edge Devices** capture traffic and perform local analysis.
2. **Edge Servers** aggregate client updates and host the UI.
3. **Cloud Server** pre-trains models and orchestrates federated rounds.

---

## Models

**Fusion Centers FL**
- **FUSION-MLP**: multi-task MLP — shared trunk + threat-class softmax head + escalation-score sigmoid head. Plain Keras variant + `FedProxFusionMLPModel` subclass with proximal-term `train_step` override.

**HFL-DNN-GAN-NIDS**
- **NIDS**: DNN binary classifier for intrusion detection.
- **Discriminator**: Multi-class classifier (real vs. synthetic).
- **Generator**: GAN-based traffic synthesizer.
- **GAN**: Combined model for adversarial data generation and classification.

---

## Documentation

> **Running the experiments? Start here:**
> **[`DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md`](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md)**
>
> The runbook is the working manual — it covers data acquisition, the
> six-experiment matrix with both single-node simulation and real
> multi-node commands, how to choose between the two modes, figure
> regeneration, and a network-debug checklist. The design and
> implementation-plan docs below are reference material; the runbook
> is what you actually need open while running experiments.

| Doc | Purpose |
|---|---|
| **[`DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md`](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md)** | **Operational runbook — start here.** Six-experiment matrix (§2), sim vs. real multi-node recommendations (§2.7), sensitive-features ablation (§3), figure regeneration (§4), real multi-node procedure with firewall checklist (§7), troubleshooting (§8). |
| [`DeveloperDocs/Fusion_Centers_FL_Update_Design.md`](DeveloperDocs/Fusion_Centers_FL_Update_Design.md) | Design reference — pipeline stages, threat model, math, as-built deviations. Read this when you need to understand *why* the pipeline looks the way it does. |
| [`DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`](DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md) | Phased build plan + per-phase outcomes, §8 review log, §9 known limitations. Read this when you need to know what was deferred, what's still untested, or why a given design decision was made. |

---

## Testing

Fusion Centers ships with 183 unit + integration tests. After install:

```bash
# Linux / macOS / WSL (and Windows PowerShell — same syntax)
pytest tests/ -q                  # full suite
pytest tests/unit -q -k fusion    # fusion-centers only
```

The `setup_fusion_node.{sh,ps1} --verify` flag runs `pytest tests/unit -k fusion`
automatically as the last step of provisioning.

---

## License

This project is licensed under the [MIT License](LICENSE).
