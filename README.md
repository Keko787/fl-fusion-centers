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
  - [Fusion Centers — Centralized Training](#fusion-centers--centralized-training)
  - [Fusion Centers — Federated Training](#fusion-centers--federated-training)
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
# Windows (PowerShell 7+)
git clone https://github.com/Keko787/fl-fusion-centers.git
cd fl-fusion-centers
pwsh -File AppSetup\setup_fusion_node.ps1 -Verify
& .\.venv\Scripts\Activate.ps1
```

Container alternative:

```bash
docker build -t fusion-centers -f AppSetup/DockerSetup/Dockerfile.fusion .
docker run --rm -v "$PWD/results:/app/results" fusion-centers \
    python App/TrainingApp/Client/TrainingClient.py --model_type FUSION-MLP --help
```

Then follow [`DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md`](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md)
for the six-experiment matrix that reproduces the paper figures.

---

## Quickstart: HFL-DNN-GAN-NIDS

```bash
# Clone the repository
git clone https://github.com/Keko787/HFL-DNN-GAN-IDS.git
cd HFL-DNN-GAN-IDS

# [Option 1] AERPAW node setup (drone-fleet testbed)
python3 AppSetup/AERPAW_node_Setup.py

# [Option 2] Chameleon node setup (CPU bare-metal)
python3 AppSetup/Chameleon_node_Setup.py
```

The NIDS stack uses `requirements_core.txt` (TF 2.15, Flower 1.9,
tensorflow-privacy 0.9.0). Don't mix it with `requirements_fusion.txt`
in the same venv — they pin incompatible TF versions.

---

## Datasets

### Fusion Centers FL — UCI Communities and Crime (Unnormalized)

Auto-fetched by `AppSetup/setup_fusion_node.{sh,ps1}`. Manual install:

```bash
mkdir -p ~/datasets/CommunitiesCrime
curl -L https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip \
    -o /tmp/comm_crime.zip
unzip -j /tmp/comm_crime.zip CommViolPredUnnormalizedData.txt -d ~/datasets/CommunitiesCrime/

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
unzip $HOME/datasets/CICIoT2023.zip -d $HOME/datasets/CICIOT2023
```

---

## Usage

All commands assume the project root is the current directory and the
appropriate venv is activated.

### Fusion Centers — Centralized Training

Single-process baseline. Sets the macro-F1 ceiling federated runs are
compared against.

```bash
python App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP --trainingArea Central \
    --partition_strategy iid --num_clients 1 --client_id 0 \
    --epochs 50 \
    --run_dir results/exp1_centralized \
    --save_name centralized_baseline
```

### Fusion Centers — Federated Training

Five-client FedAvg over a geographic non-IID partition (the realistic
fusion-center scenario):

```bash
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --fl_strategy FedAvg \
    --partition_strategy geographic \
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 \
    --run_dir results/exp4_fedavg_geo \
    --save_name fedavg_geo_n5
```

FedProx variant (drift-robust on non-IID partitions):

```bash
python App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --fl_strategy FedProx --fedprox_mu 0.01 \
    --partition_strategy geographic \
    --num_clients 5 --rounds 100 --epochs 1 --min_clients 5 \
    --run_dir results/exp5_fedprox_geo \
    --save_name fedprox_geo_n5
```

Full experiment matrix (including local-only baseline, IID FedAvg, and
the N ∈ {3, 5, 10} scaling sweep) is in
[`RUNNING_FUSION_EXPERIMENTS.md`](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md).

### NIDS — Federated Training (Host)

```bash
python3 App/TrainingApp/HFLHost/HFLHost.py --help
```

### NIDS — Localized & Federated Training (Client)

```bash
python3 App/TrainingApp/Client/TrainingClient.py --help
# Default uses CICIOT2023 dataset; use --dataset IOTBOTNET for IoTBotnet.
```

Demo of localized NIDS training:

```bash
python3 App/TrainingApp/Client/TrainingClient.py \
    --model_type AC-GAN --model_training Both --trainingArea Central \
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

| Doc | Purpose |
|---|---|
| [`DeveloperDocs/Fusion_Centers_FL_Update_Design.md`](DeveloperDocs/Fusion_Centers_FL_Update_Design.md) | Design document — pipeline stages, threat model, math, as-built deviations |
| [`DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`](DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md) | Phased plan + per-phase build outcomes, §8 review log, §9 known limitations |
| [`DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md`](DeveloperDocs/RUNNING_FUSION_EXPERIMENTS.md) | Operational runbook — six-experiment matrix, ablations, figure regeneration, troubleshooting |

---

## Testing

Fusion Centers ships with 183 unit + integration tests. After install:

```bash
pytest tests/ -q                  # full suite
pytest tests/unit -q -k fusion    # fusion-centers only
```

The `setup_fusion_node.{sh,ps1} --verify` flag runs `pytest tests/unit -k fusion`
automatically as the last step of provisioning.

---

## License

This project is licensed under the [MIT License](LICENSE).
