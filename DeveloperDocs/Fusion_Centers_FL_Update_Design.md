# Fusion Centers FL Update — Design Document

Project: Federated Predictive Threat Intelligence for Cross-Jurisdictional Fusion Centers
Status: **All build phases (0, A, B, C, D, E) closed (2026-05-11) — 183 tests passing.** Awaiting first real-data shakedown run. See §10 for as-built deviations.
Companion: [Fusion_Centers_FL_Update_Implementation_Plan.md](Fusion_Centers_FL_Update_Implementation_Plan.md) — phased build with review log + per-phase outcomes. Operational runbook: [RUNNING_FUSION_EXPERIMENTS.md](RUNNING_FUSION_EXPERIMENTS.md).
Author: see `Project_Outline_FL_Fusion_Centers.docx`

---

## 1. Goal

Re-target the existing **HFL-DNN-GAN-NIDS** framework (IoT intrusion detection, hierarchical FL, GANs, TensorFlow + Flower) at a **flat federated, multi-task MLP** for cross-jurisdictional crime/threat-intelligence prediction on the **UCI Communities and Crime (Unnormalized)** dataset, runnable on a single Chameleon Cloud bare-metal node via Flower simulation.

The update reuses the existing config-driven training framework as-is and adds a new dataset, new model family, new training-config classes, and new federation strategies in the same drop-in pattern the codebase already uses for `NIDS`, `GAN`, `WGAN-GP`, `AC-GAN`, `CANGAN`. **No existing model path is altered.**

## 2. Existing Framework — Anatomy We Are Building On

The training pipeline is a five-stage, argument-driven flow. Both [TrainingClient.py](App/TrainingApp/Client/TrainingClient.py) and [HFLHost.py](App/TrainingApp/HFLHost/HFLHost.py) follow it identically.

| Stage | Module | Role |
|---|---|---|
| 1. Args | [ArgumentConfigLoad.py](Config/SessionConfig/ArgumentConfigLoad.py) | `parse_training_client_args()`, `parse_HFL_Host_args()` — define choices + computed flags |
| 2. Data | [datasetLoadProcess.py](Config/SessionConfig/datasetLoadProcess.py) | Switch on `args.dataset` → loader; switch on `args.dataset_processing` → preprocessor |
| 3. Hyperparams | [hyperparameterLoading.py](Config/SessionConfig/hyperparameterLoading.py) | Switch on `args.model_type` → returns the same fat tuple consumed by all trainers |
| 4. Model | [modelCreateLoad.py](Config/SessionConfig/modelCreateLoad.py) | Switch on `modelType` × `train_type` → returns `(nids, discriminator, generator, GAN)` |
| 5. Trainer | [modelCentralTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/modelCentralTrainingConfigLoad.py) / [modelFederatedTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/modelFederatedTrainingConfigLoad.py) | Switch on model_type × train_type → instantiates a trainer class implementing `.fit() / .evaluate() / .save()` (Flower `NumPyClient` for federated) |

Server-side strategy is selected in [HFLStrategyTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/HFLStrategyTrainingConfigLoad.py) (`_run_standard_federation_strategies`, `_run_fit_on_end_strategies`).

**Key invariant:** every new feature is a new branch in the existing switches, plus a new file under `Config/`. Entry-point scripts only learn one new `--model_type` value.

The framework is currently **TensorFlow/Keras + Flower**. The fusion-center outline specifies **PyTorch + Flower**. Section 8 covers the backend decision.

## 3. What Changes — Mapped Stage-by-Stage

### 3.1 Stage 1 — Arguments

Add to `ArgumentConfigLoad.py`:

- **Dataset choice**: extend `--dataset` choices with `"COMMCRIME"` (and `"NIBRS"` reserved as the optional secondary dataset, gated behind a not-yet-implemented loader).
- **Dataset processing**: extend `--dataset_processing` choices with `"COMMCRIME"` (default scaling + label-engineering pipeline) and `"COMMCRIME-MM"` (MinMax variant for ablations).
- **Model type**: extend `--model_type` choices with `"FUSION-MLP"` (multi-task MLP — the new model family).
- **Model training**: extend `--model_training` choices with `"MultiTask"` (the only sub-mode for `FUSION-MLP` — both heads always train together).
- **New fusion-specific args** (grouped under a `─── Fusion Center / Communities-Crime Settings ───` block to mirror the existing CICIOT block):
  - `--num_clients` (int, default 5; choices 3, 5, 10) — number of simulated agencies for Flower simulation.
  - `--partition_strategy` (`"geographic" | "iid" | "dirichlet"`, default `"geographic"`) — how to split across clients.
  - `--dirichlet_alpha` (float, default 0.5) — only used when `partition_strategy == "dirichlet"`.
  - `--client_id` (int, default 0) — which partition this client process loads (only used in real multi-process runs; ignored in simulation).
  - `--global_test_size` (float, default 0.15) — held-out global test split.
  - `--escalation_loss_weight` (float, default 0.5) — `β` in `L = α·CE + β·BCE`; `α` is `1 - β` (clamped).
  - `--drop_sensitive_features` (store_true) — drop documented-bias columns (race, ethnicity, income breakdowns).
- **Federation strategy** (server-side, in `parse_HFL_Host_args`):
  - Extend with `--fl_strategy` choices `"FedAvg" | "FedProx"` (default `FedAvg`).
  - `--fedprox_mu` (float, default 0.01) — proximal term coefficient when strategy is FedProx.
- **Conditional logic** in the post-parse block (mirrors the existing AC-GAN auto-set):
  ```python
  if args.model_type == "FUSION-MLP":
      args.model_training = "MultiTask"
      args.dataset = "COMMCRIME"
      if args.dataset_processing == "Default":
          args.dataset_processing = "COMMCRIME"
  ```

### 3.2 Stage 2 — Dataset Loading & Preprocessing

Add a new dataset loader, mirroring [CICIOT2023_Sampling/](Config/DatasetConfig/CICIOT2023_Sampling/):

```
Config/DatasetConfig/CommunitiesCrime_Sampling/
    commCrimeDatasetLoad.py      # Phases 1–3 of outline §6: download/parse, audit, clean
    commCrimeLabelEngineering.py # Phase 4: derive threat_class + escalation_score
    commCrimeFederatedPartition.py # Phase 7: state→region bucketing, IID, Dirichlet
```

The existing `Config/DatasetConfig/Dataset_Preprocessing/datasetPreprocess.py` gets a new function `preprocess_communities_crime(...)` that handles Phase 5 (scaling, encoding) and Phase 6 (global test split). It returns the **same shape** the existing pipeline expects:

`X_train, X_val, y_train, y_val, X_test, y_test`

with one extension — `y_*` is a **tuple `(y_class, y_escalation)`** instead of a single array. Existing trainers ignore this (they branch by `model_type`), so no collision. The new `FUSION-MLP` trainer is the only consumer that unpacks the tuple.

`datasetLoadProcess.py` gets two new branches:

```python
elif dataset_used == "COMMCRIME":
    raw = loadCommunitiesCrime(
        path=args.commcrime_path,
        drop_sensitive=args.drop_sensitive_features,
    )
    # Phases 4 + 7 done up front so per-client partitioning is deterministic
    labeled = engineerCommCrimeLabels(raw)
    partitions = partitionCommCrime(
        labeled,
        strategy=args.partition_strategy,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.commcrime_random_seed,
        global_test_size=args.global_test_size,
    )
    selected = partitions[args.client_id]  # one client's slice
    return preprocess_communities_crime(selected, args.dataset_processing)
```

**Reproducibility artifacts** written under `results/commcrime_run_<timestamp>/`:
- `partitions/client_<i>.parquet`, `global_test.parquet`
- `scaler.joblib`
- `partition_stats.json` (per-client class distribution, sample counts, feature mean/std — figures for the paper)

### 3.3 Stage 3 — Hyperparameters

Add a `FUSION-MLP` branch to `hyperparameterLoading.py` returning the same 20-tuple. Most GAN-specific slots are `None`:

| Slot | FUSION-MLP value |
|---|---|
| `BATCH_SIZE` | 128 |
| `noise_dim` | None |
| `steps_per_epoch` | `len(X_train_data) // BATCH_SIZE` |
| `input_dim` | `X_train_data.shape[1]` |
| `num_classes` | 3 (violent / property / other) |
| `latent_dim` | None |
| `betas` | `[0.9, 0.999]` |
| `learning_rate` | 1e-3 |
| `l2_alpha` | 1e-4 (only if `regularizationEnabled`) |
| early-stop / lr-sched / checkpoint slots | reuse existing logic, monitor `val_loss` |

The escalation-loss weight (`α`, `β`) and joint-loss weighting are passed via `args` directly into the trainer rather than crammed into this tuple, since no existing path uses them — keeps the tuple stable for everything else.

### 3.4 Stage 4 — Model Creation

Add to `Config/modelStructures/`:

```
Config/modelStructures/FusionMLP/
    multiTaskMLPStruct.py    # build_fusion_mlp(input_dim, num_classes, hidden=(128,64,32), dropout=0.2)
```

The model has the shared trunk (3 dense + ReLU + dropout + batchnorm) and two heads (softmax over `num_classes`, sigmoid scalar). It is the only object returned to the rest of the pipeline.

`modelCreateLoad.py` gets a new branch returning the model in the existing 4-tuple slot:

```python
elif modelType == "FUSION-MLP":
    if pretrainedNids:                      # reuse the existing nids slot for the multi-task model
        model = load_fusion_mlp(pretrainedNids)
    else:
        model = build_fusion_mlp(input_dim, num_classes)
    return model, None, None, None          # (nids, discriminator, generator, GAN)
```

This places the multi-task MLP in the `nids` slot, which is the natural fit — it is the discriminative head of the system and trainers downstream already treat that slot as "the main classifier."

### 3.5 Stage 5 — Training Config Classes

Mirror the existing folder layout exactly. Two new classes — one centralized, one federated.

```
Config/ModelTrainingConfig/ClientModelTrainingConfig/
    CentralTrainingConfig/FusionMLP/
        fusionMLPCentralTrainingConfig.py     # CentralFusionMLPClient
    HFLClientModelTrainingConfig/FusionMLP/
        fusionMLPClientConfig.py              # FlFusionMLPClient (Flower NumPyClient)
```

Both implement the same trainer contract used by every existing config:

| Method | Behavior |
|---|---|
| `__init__(model, X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE, epochs, steps_per_epoch, learning_rate, escalation_weight, ...)` | Compile model with **joint loss** `α·CrossEntropy + β·BCE` and dual metrics |
| `.fit()` (central) / `.fit(parameters, config)` (federated) | One round of local training |
| `.evaluate()` / `.evaluate(parameters, config)` | Reports macro-F1, per-class P/R, escalation MAE, escalation AUROC |
| `.save(save_name)` | Saves to `ModelArchive/fusion_mlp_<save_name>.{h5\|pt}` |
| `.recordTraining()` / `.recordEvaluation()` | Same log format as `FlNidsClient` (Node, Round, metrics) so downstream `Analysis/TestAnalysis` keeps working |

Wire-in:

- `modelCentralTrainingConfigLoad.py` gets `elif model_type == "FUSION-MLP": client = CentralFusionMLPClient(...)`
- `modelFederatedTrainingConfigLoad.py` gets the matching `FlFusionMLPClient(...)` branch
- Both forward `args.escalation_loss_weight` (and any new fusion-specific args). Pass via `args` rather than expanding the existing positional tuple, to avoid disturbing every other trainer signature.

### 3.6 Server-Side — Flat Federation + FedProx

The current `HFLStrategyTrainingConfigLoad.py` only wires `FedAvg`. Add a `FedProx` strategy plus a "fusion-centers default" branch:

```
Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/
    FusionFedAvgConfig.py    # thin wrapper around fl.server.strategy.FedAvg with eval logging
    FusionFedProxConfig.py   # implements FedProx (proximal mu)
    SimulationRunner.py      # entry for fl.simulation.start_simulation (single-node N-client)
```

`HFLHost.py` gains one branch up-front:

```python
if args.model_type == "FUSION-MLP":
    if args.fl_strategy == "FedProx":
        run_fusion_fedprox(args)
    else:
        run_fusion_fedavg(args)
    return
```

Single-node simulation mode (the deployment target per outline §7.2) is the default for the fusion experiments. `SimulationRunner.py` calls `fl.simulation.start_simulation(client_fn=..., num_clients=args.num_clients, ...)`. Multi-process / multi-node deployment continues to work via the existing `TrainingClient.py` + `--host` path with no changes.

`min_fit_clients`, `min_evaluate_clients`, `min_available_clients` already exist on the host args and map straight onto the new strategies.

## 4. New Files Summary

```
Config/
├── DatasetConfig/
│   └── CommunitiesCrime_Sampling/
│       ├── commCrimeDatasetLoad.py
│       ├── commCrimeLabelEngineering.py
│       └── commCrimeFederatedPartition.py
├── modelStructures/
│   └── FusionMLP/
│       └── multiTaskMLPStruct.py
└── ModelTrainingConfig/
    ├── ClientModelTrainingConfig/
    │   ├── CentralTrainingConfig/FusionMLP/fusionMLPCentralTrainingConfig.py
    │   └── HFLClientModelTrainingConfig/FusionMLP/fusionMLPClientConfig.py
    └── HostModelTrainingConfig/FusionCenters/
        ├── FusionFedAvgConfig.py
        ├── FusionFedProxConfig.py
        └── SimulationRunner.py
```

Edits (additive, no removals):
- `Config/SessionConfig/ArgumentConfigLoad.py` — new args + post-parse conditional
- `Config/SessionConfig/datasetLoadProcess.py` — `COMMCRIME` branch
- `Config/SessionConfig/Dataset_Preprocessing/datasetPreprocess.py` — `preprocess_communities_crime`
- `Config/SessionConfig/hyperparameterLoading.py` — `FUSION-MLP` branch
- `Config/SessionConfig/modelCreateLoad.py` — `FUSION-MLP` branch
- `Config/SessionConfig/ModelTrainingConfigLoad/modelCentralTrainingConfigLoad.py` — branch
- `Config/SessionConfig/ModelTrainingConfigLoad/modelFederatedTrainingConfigLoad.py` — branch
- `App/TrainingApp/HFLHost/HFLHost.py` — early `FUSION-MLP` dispatch to `SimulationRunner`

## 5. Experiment Matrix → Existing Framework Mapping

The outline §7.4 matrix, expressed as command lines on the existing entry points:

| # | Experiment | Command |
|---|---|---|
| 1 | Centralized baseline | `TrainingClient.py --model_type FUSION-MLP --trainingArea Central --partition_strategy iid --num_clients 1` |
| 2 | Local-only (per client, no FL) | `TrainingClient.py --model_type FUSION-MLP --trainingArea Central --partition_strategy geographic --num_clients 5 --client_id <0..4>` |
| 3 | FedAvg / IID | `HFLHost.py --model_type FUSION-MLP --fl_strategy FedAvg --partition_strategy iid --num_clients 5 --rounds 100` |
| 4 | FedAvg / non-IID geographic | `HFLHost.py --model_type FUSION-MLP --fl_strategy FedAvg --partition_strategy geographic --num_clients 5 --rounds 100` |
| 5 | FedProx / non-IID geographic | `HFLHost.py --model_type FUSION-MLP --fl_strategy FedProx --fedprox_mu 0.01 --partition_strategy geographic --num_clients 5 --rounds 100` |
| 6 | Scaling — N=3, 5, 10 | Loop #4/#5 over `--num_clients {3,5,10}` |

Convergence stop (100 rounds OR 10-round plateau) is implemented inside the new strategy classes. All six configurations write to distinct directories under `results/commcrime_run_<timestamp>/<config_name>/`.

## 6. Metrics & Logging

Reuse the existing `recordTraining` / `recordEvaluation` text-log format so the existing `Analysis/TestAnalysis` consumers keep working. Add to the evaluation log:

- `class_macro_f1`, `class_per_class_precision`, `class_per_class_recall` (Phase 7.5 outline metrics)
- `escalation_mae`, `escalation_auroc`
- `parameter_update_wire_bytes`, `parameter_update_wire_bytes_per_client`, `round_seconds`, `proximal_contribution` (federation-overhead + FedProx contribution metrics)
- `per_client_test_accuracy_variance` (fairness)

A new `Analysis/CommunitiesCrime/` folder can host the plot scripts in a follow-up; it is not blocking.

## 7. Reproducibility

Driven by existing patterns + outline §6.9:

- Every stochastic step seeded — extend `ciciot_random_seed` pattern with `commcrime_random_seed` (default 42).
- Fitted scaler and global test split written once and re-loaded on subsequent runs (idempotent loader).
- Per-client partition stats written to disk for every run, regardless of strategy, so paper figures regenerate from logs.
- `pip freeze` snapshot dumped alongside each run's results directory (mirrors the AERPAW / Chameleon provisioning convention already used in `AppSetup/`).

## 8. Open Decisions — all closed at Phase 0 (2026-05-10)

| Decision | Outcome | Where it landed |
|---|---|---|
| **TF/Keras vs PyTorch** | TF/Keras (option A) | Zero churn for the existing pipeline; Flower stays backend-agnostic. All `[backend-fork]` notes in the implementation plan are now inert. |
| **NIBRS extension** | Reserved | `--dataset NIBRS` choice exists; `datasetLoadProcess.py` raises `NotImplementedError`. Contract locked, work deferrable. |
| **Hermes mode interaction** | Reject | `SystemExit("FUSION-MLP does not support --mode hermes; use --mode legacy.")` in both `TrainingClient.py` / `HFLHost.py` *before* the legacy hermes dispatch, plus defense-in-depth in `load_commcrime`. |
| **Sensitive-feature drop default** | `True` | `args.drop_sensitive_features` defaults to True via `argparse.BooleanOptionalAction`. Override with `--no-drop_sensitive_features` for the Phase E ablation row. The actually-dropped column list is recorded per-run in `partition_stats.json["dropped_sensitive_columns"]`. |

## 9. Phasing

A reasonable build order, roughly one PR per phase:

1. **Phase A — Data path.** Loader, label engineering, partitioner, preprocessing, partition-stats artifacts. Validates against a local CSV; no model training yet.
2. **Phase B — Centralized FUSION-MLP.** Model struct, hyperparam branch, central trainer, end-to-end run via `TrainingClient.py --trainingArea Central`. Produces baseline numbers.
3. **Phase C — Federated FUSION-MLP via FedAvg.** Client trainer + `SimulationRunner` for single-node simulation. Gives experiments 3 and 4.
4. **Phase D — FedProx + scaling.** Strategy class + `--num_clients` sweep. Gives experiments 5 and 6.
5. **Phase E — Paper-ready logging + ablations.** `--drop_sensitive_features` ablation, per-client fairness reporting, plot scripts under `Analysis/CommunitiesCrime/`.

Each phase is independently shippable and leaves all existing IoT/GAN paths untouched.

**Phase outcomes (2026-05-11):** A ✅ (63 tests at close), B ✅ (90), C ✅ (119), D ✅ (146), E ✅ (183). All phases closed; six code-review passes applied (Phase E got two). See [Implementation Plan §8](Fusion_Centers_FL_Update_Implementation_Plan.md) for the per-phase review logs.

---

## 10. As-Built Deviations from the Design

The Phases A–D builds matched the design above except for the items below — each driven by either a code-review pass or an implementation detail that emerged during the build. The implementation plan §8 captures the specific reviews that motivated each change.

### 10.1 New modules beyond the §4 inventory

The design listed 8 new modules; the implementation shipped 11. The extras emerged from refactoring:

| Module | Why it exists |
|---|---|
| [Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimePreprocess.py](Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimePreprocess.py) | Phase A: keeps the preprocessing math TF-free so the data-path tests don't transitively import `flwr`/`tensorflow`. Also hosts `fit_global_scaler` (Phase A.7 follow-up for the Phase C "fit once on union" contract). |
| [Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimeLoadProcess.py](Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimeLoadProcess.py) | Phase A.5 + Phase C: `load_commcrime` (single client) and `load_commcrime_for_simulation` (all clients) share the download → clean → label → partition pipeline via the private `_prepare_partitions` helper. Single source of truth for hermes-mode rejection + run-dir resolution + reproducibility artifact write-out. |
| [Config/SessionConfig/runArtifacts.py](Config/SessionConfig/runArtifacts.py) | Phase A.7 — reproducibility helpers: `init_run_dir`, `resolve_run_dir` (added Phase C review #1 for `--run_dir` sharing), `dump_pip_freeze`, `seed_all`. |

### 10.2 On-disk format changes

| Spec'd format | Shipped format | Why |
|---|---|---|
| `partitions/client_<i>.parquet` | `.pkl` | Zero new dependencies; `pickle` round-trips pandas DataFrames natively. The Chameleon provisioning script doesn't gain `pyarrow`. |
| `ModelArchive/...h5` | `<run_dir>/...keras` | Keras 3 deprecation; `.keras` zip format is the new recommended path. The FUSION-MLP runs also park the model artifact inside the run dir (alongside scaler + logs) rather than the project-level `ModelArchive/`. |
| Logs in `evaluationLog`/`trainingLog` argparse strings | Same args, but argparse defaults to distinct `{timestamp}_{evaluation,training}.log` filenames, and `load_commcrime` rewrites both paths to live inside `<run_dir>/` | Phase B review caught that the legacy project pattern collapsed training + eval logs into a single bare-timestamp file; FUSION-MLP runs need distinct logs for Phase E plot scripts. |

### 10.3 FedProx implementation specifics

The design (§3.6) was light on the FedProx model wrapping. The implementation:

* Introduces `FedProxFusionMLPModel(tf.keras.Model)` — a Functional-API subclass that overrides `train_step` to add `(μ/2)·Σ‖w-g‖²` to the loss. Gated on `μ>0` AND a prior `set_global_weights(...)` call so the model behaves identically to the base when either is absent.
* `set_global_weights(weights)` takes the full Flower weights list (length matches `model.get_weights()`) and filters internally to the trainable subset via a cached `_trainable_indices` mask. Phase D review #1 caught a real shape-mismatch crash that the planning step under-specified.
* Anchor variables stored via `self.__dict__["..."]` to suppress Keras's variable tracker — keeps `model.weights` clean so the `.keras` artifact doesn't bloat. Phase D review #6 added a test.
* `fedprox_mu` is a `tf.Variable`, not a Python attr — Phase D review #2 caught that Python-side conditions inside `train_step` get frozen at trace time; the variable + `tf.where` ensures per-round μ schedules would behave correctly if added.
* The model emits a sibling `proximal_contribution` metric (post-review follow-up). `history.history["loss"]` excludes the proximal term per Keras's `compute_loss` contract — `proximal_contribution` is the way users see what the FedProx penalty is doing.

### 10.4 Strategy class details

* `FusionFedAvg` (§3.6 spec'd as "thin wrapper") gained substantial behavior:
  * Overrides `aggregate_fit` to stash `self._final_parameters` (Phase C review #1 — without this the saved model is byte-identical to the initial weights).
  * `aggregate_evaluate` weighted-means with **NaN filtering** (Phase C review #2 — single-client AUROC=NaN no longer drags aggregated values toward zero).
  * Per-client F1 variance as the fairness metric.
  * Plateau detection on aggregated loss (logs a warning when fired; full simulation still completes the round budget — early termination deferred).
* `FusionFedProx(FusionFedAvg)` (§3.6 spec'd separately) is now a 30-LOC subclass that inherits all of the above and overrides only `configure_fit` to broadcast `μ` in the per-round config dict.

### 10.5 Simulation runner contract

* Single source of truth for hyperparameters: `run_fusion_simulation` calls `hyperparameterLoading` once on a representative client's `X_train` (Phase D review #4) rather than hardcoding BATCH_SIZE / lr / num_classes / l2_alpha as it did initially.
* Per-cid client instance cache in `_build_client_fn`'s closure (Phase D review #7) — model built once per client, reused across rounds.
* Honors `args.regularizationEnabled` (Phase D review #5) — l2_alpha is zeroed out if the user disables regularization.

### 10.6 Test coverage details (vs §6 spec'd "evaluation log" only)

| Test pyramid layer | Spec count | Built count |
|---|---|---|
| Unit | "Label-engineering math, partitioner determinism, scaler persistence, model architecture, joint-loss, FedProx proximal-term math" | **10 files / ~80 tests** including the same plus: anchor-no-leak test, length-validation test, FedProx end-to-end through FL client, strategy aggregation including NaN handling, reproducibility tests, `proximal_contribution` metric tests |
| Integration | "Per-phase smoke runs on synthetic 100-row partitions" | **6 files / ~66 tests** including dispatcher integration (Phase A), central training (Phase B), federated simulation construction (Phase C), N=3/5/10 scaling sweep (Phase D) |
| **Total** | — | **146 tests passing** in ~76 s |

### 10.7 Sensitive-features list (§5.3)

The design left the exact column list open. Shipped:

* **Race/ethnicity share (4):** `racepctblack`, `racePctWhite`, `racePctAsian`, `racePctHisp`
* **Per-capita income by race (6):** `whitePerCap`, `blackPerCap`, `indianPerCap`, `AsianPerCap`, `OtherPerCap`, `HispPerCap`
* **Aggregate income / public-assistance (4):** `medIncome`, `perCapInc`, `pctWInvInc`, `pctWPubAsst`

Rationale: race/ethnicity share is the obvious set; per-capita-income-by-race buckets are racial proxies; aggregate income is included because outline §5.3 calls out "race, ethnicity, **income**" together and predictive-policing literature documents aggregate-income features as proxies for protected attributes. The actually-dropped list (intersection with the cleaned DataFrame) is persisted per-run in `partition_stats.json["dropped_sensitive_columns"]` so the Phase E ablation has authoritative comparison data.

### 10.8 Phase E deliverables — all shipped

| Item | Where | Notes |
|---|---|---|
| Federation-overhead metrics (`round_seconds`, `parameter_update_wire_bytes`, `parameter_update_wire_bytes_per_client`, `proximal_contribution`) | `FusionFedAvg.configure_fit` stamps the start time; `aggregate_fit` computes byte sum + duration + weighted-mean proximal | Folded into `aggregate_evaluate`'s log row so the plot scripts read everything from one place. Wire-bytes is the post-serialization payload — federation-overhead measurement, not raw float32 weight bytes |
| Fairness accuracy variance + `threat_accuracy` per-client metric | `_compute_metrics` (Phase B trainer) + `FusionFedAvg.aggregate_evaluate` | Both fairness measures (`fairness_macro_f1_variance`, `fairness_accuracy_variance`) emitted per round |
| `Analysis/CommunitiesCrime/log_parser.py` | `parse_server_log` / `parse_client_log` / `parse_partition_stats` / `collect_server_logs` | Pure stdlib + pandas; no ML deps; ~10 s test suite |
| `plot_per_client_distribution.py` | Outline §6.7 figure | CLI + `plot(...)` function form |
| `plot_centralized_vs_federated.py` | Outline §7.9 headline figure | Multi-line convergence with optional centralized baseline reference |
| `plot_scaling_n_clients.py` | Outline §7.4 row 6 | Twin-axis: final macro-F1 bars + rounds-to-convergence line |
| `RUNNING_FUSION_EXPERIMENTS.md` | `DeveloperDocs/` | 6-experiment matrix + ablation section + figure regeneration + troubleshooting |

### 10.9 Items deferred beyond Phase E

* **`start_simulation` → `run_simulation` migration** — Flower 1.29 still supports the legacy API with a deprecation warning. Migration deferred until either Flower drops it or an experiment needs the new `ServerAppComponents` features.
* **NIBRS loader** — `--dataset NIBRS` choice exists and raises `NotImplementedError`. Optional outline §5.2 extension; deferrable.
* **Ray-backed end-to-end simulation test** — current integration tests cover building blocks separately; `fl.simulation.start_simulation` itself is invoked only via real runs. See [§9.4](Fusion_Centers_FL_Update_Implementation_Plan.md#94--flsimulationstart_simulation-never-end-to-end-tested) of the implementation plan.

### 10.9 Known limitations + untested boundaries

The 146-test suite does **not** validate the dataset URL, the `.names` file format, full Ray-backed simulation, the real-data DoD numbers from each phase, or the existing-IoT/GAN regression suite in the current Windows + TF 2.21 environment. The implementation plan [§9](Fusion_Centers_FL_Update_Implementation_Plan.md#9-known-limitations--untested-boundaries) enumerates each gap with severity, what triggers it, and the suggested first-real-run mitigation. **Read §9 before scheduling the first real UCI download + simulation.**
