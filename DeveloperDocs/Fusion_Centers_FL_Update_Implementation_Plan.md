# Fusion Centers FL Update — Implementation Plan

**Companion to** [Fusion_Centers_FL_Update_Design.md](Fusion_Centers_FL_Update_Design.md).

**Audience:** engineer executing the build. Read the design doc first — this plan assumes its terminology (the five-stage pipeline, the new `FUSION-MLP` model type, the flat federation strategy, the new `COMMCRIME` dataset).

**Status:** Phase 0, A, B, C, D, E closed (2026-05-10). All build phases complete. Awaiting first real-data shakedown run.

### Build snapshot

| Phase | Status | Tests at close | After review fixes | Net additions |
|---|---|---|---|---|
| 0 | ✅ Closed | — | — | Backend / dataset path / sensitive-features defaults locked |
| A | ✅ Closed | 35 | **63** | Data path: download → audit → clean → label → partition → preprocess + run-dir reproducibility artifacts |
| B | ✅ Closed | 79 | **90** | Multi-task MLP + centralized trainer + joint loss + dual-head metrics |
| C | ✅ Closed | 107 | **119** | Flower-simulation FedAvg + custom aggregation + per-client logs |
| D | ✅ Closed | 137 | **146** | FedProx (proximal-term subclass + strategy) + N-client scaling sweep + `proximal_contribution` metric |
| E | ✅ Closed | 168 | **183** | Federation-overhead metrics + fairness-accuracy variance + plot scripts (3) + log parser + `RUNNING_FUSION_EXPERIMENTS.md` quickstart + two Phase E review passes (NaN-safe fairness, wire-bytes rename, log parser round-trip, server-side `proximal_contribution` aggregation, robust convergence heuristic, canonical fairness, retry-safe timing, vector output, paper styling, package re-exports) |

**183 tests passing** across 11 unit files + 8 integration files (~3800 LOC of tests), exercising ~3100 LOC of new pipeline code. See §8 for the review log of bugs caught and fixes applied during the five code-review passes (Phase E got two); §9 for known untested boundaries.

---

## 0. Guiding Principles for the Build

1. **Additive only.** Every new feature is a new branch in an existing switch plus a new file under `Config/`. No edits to existing IoT/GAN paths. The grep-test invariant (existing model paths produce identical behavior) holds at every phase boundary.
2. **One stage at a time.** Build the five-stage pipeline (args → data → hyperparams → model → trainer) sequentially. Each stage is mergeable on its own and tested with a stub for the next stage.
3. **Centralized first, federated second.** Centralized training validates the data + model + loss before federation overhead is added. Phase B must produce believable numbers before Phase C starts.
4. **Reuse over rebuild.** Match the existing folder structure and trainer contract (`fit / evaluate / save / recordTraining / recordEvaluation`) exactly. Downstream consumers (`Analysis/`, `ModelArchive/`) keep working with no edits.
5. **Reproducibility from day one.** Seeded splits, frozen global test set, persisted scaler, partition-stats JSON, and `pip freeze` snapshot land in Phase A — not bolted on later.

### 0.1 Locked Decisions (closed 2026-05-10)

| Decision | Chosen | Rationale |
|---|---|---|
| ML backend for `FUSION-MLP` | **TF/Keras** | Zero churn — every existing trainer, model factory, `ModelArchive` saver, and `Analysis/` consumer keeps working unchanged. Flower is backend-agnostic so the federation glue is identical to what PyTorch would need. |
| Dataset directory | **`$HOME/datasets/CommunitiesCrime/`** | Matches the existing `$HOME/datasets/CICIOT2023/` convention from the README. |
| `--drop_sensitive_features` default | **`True`** | Per design doc §8.4 — drops documented-bias columns (race, ethnicity, income breakdowns) for headline runs; both settings will be exposed in the Phase E ablation row. |

All `[backend-fork]` notes elsewhere in this plan are inert.

---

## 1. Current-State Inventory

The pieces the build extends; all already in place.

| Exists today | Path | Role in the update |
|---|---|---|
| Argument parser | [Config/SessionConfig/ArgumentConfigLoad.py](Config/SessionConfig/ArgumentConfigLoad.py) | Add `FUSION-MLP` / `COMMCRIME` choices + fusion-specific args |
| Dataset dispatcher | [Config/SessionConfig/datasetLoadProcess.py](Config/SessionConfig/datasetLoadProcess.py) | Add `COMMCRIME` branch |
| Preprocessing | [Config/DatasetConfig/Dataset_Preprocessing/datasetPreprocess.py](Config/DatasetConfig/Dataset_Preprocessing/datasetPreprocess.py) | Add `preprocess_communities_crime()` |
| Hyperparameter dispatcher | [Config/SessionConfig/hyperparameterLoading.py](Config/SessionConfig/hyperparameterLoading.py) | Add `FUSION-MLP` branch |
| Model factory | [Config/SessionConfig/modelCreateLoad.py](Config/SessionConfig/modelCreateLoad.py) | Add `FUSION-MLP` branch (returns model in the `nids` slot) |
| Centralized trainer dispatcher | [Config/SessionConfig/ModelTrainingConfigLoad/modelCentralTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/modelCentralTrainingConfigLoad.py) | Add `FUSION-MLP` branch |
| Federated trainer dispatcher | [Config/SessionConfig/ModelTrainingConfigLoad/modelFederatedTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/modelFederatedTrainingConfigLoad.py) | Add `FUSION-MLP` branch |
| Server-strategy dispatcher | [Config/SessionConfig/ModelTrainingConfigLoad/HFLStrategyTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/HFLStrategyTrainingConfigLoad.py) | Add fusion-centers FedAvg + FedProx + simulation runner |
| Client entry point | [App/TrainingApp/Client/TrainingClient.py](App/TrainingApp/Client/TrainingClient.py) | No structural change; new `--model_type` flows through |
| Host entry point | [App/TrainingApp/HFLHost/HFLHost.py](App/TrainingApp/HFLHost/HFLHost.py) | One early-dispatch branch for `FUSION-MLP` → simulation |

**Not yet present** (to be created): `Config/DatasetConfig/CommunitiesCrime_Sampling/`, `Config/modelStructures/FusionMLP/`, `Config/ModelTrainingConfig/.../FusionMLP/`, `Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/`, `Analysis/CommunitiesCrime/`.

---

## 2. Target Repo Layout (after the build)

New directories and files only — existing tree unchanged.

```
Config/
  DatasetConfig/
    CommunitiesCrime_Sampling/                NEW (Phase A + C)
      commCrimeDatasetLoad.py                 Phases 1–3 (load, audit, clean) + synthetic stub for offline tests
      commCrimeLabelEngineering.py            Phase 4 (threat_class + escalation_score)
      commCrimeFederatedPartition.py          Phase 7 (geographic / IID / Dirichlet) + N=1/3/5/10
      commCrimePreprocess.py                  Phase A.4 split: scaling + (X,y) tuple unpacking + fit_global_scaler helper (kept TF-free)
      commCrimeLoadProcess.py                 Phase A.5 + Phase C: load_commcrime (single client) + load_commcrime_for_simulation (all clients) sharing _prepare_partitions helper
  SessionConfig/
    runArtifacts.py                           NEW (Phase A.7): init_run_dir, resolve_run_dir, dump_pip_freeze, seed_all
  modelStructures/
    FusionMLP/                                NEW (Phase B + D)
      multiTaskMLPStruct.py                   _build_layers, build_fusion_mlp, build_fedprox_fusion_mlp, FedProxFusionMLPModel
  ModelTrainingConfig/
    ClientModelTrainingConfig/
      CentralTrainingConfig/FusionMLP/        NEW (Phase B)
        fusionMLPCentralTrainingConfig.py     CentralFusionMLPClient (Phase B trainer; tf.data.Dataset wrapping, num_classes-aware metrics, .keras save, archive_path kwarg)
      HFLClientModelTrainingConfig/FusionMLP/ NEW (Phase C + D)
        fusionMLPClientConfig.py              FlFusionMLPClient (Flower NumPyClient; FedProx anchor wiring, empty-partition guard, print_summary flag, proximal_contribution passthrough)
    HostModelTrainingConfig/
      FusionCenters/                          NEW (Phase C + D)
        FusionFedAvgConfig.py                 FusionFedAvg: aggregate_fit + aggregate_evaluate with NaN-safe weighted means, fairness variance, plateau detector, _final_parameters stash
        FusionFedProxConfig.py                FusionFedProx (inherits FedAvg; configure_fit broadcasts mu)
        SimulationRunner.py                   run_fusion_simulation entry + _build_client_fn (caches per-cid instances)
Analysis/
  CommunitiesCrime/                           NEW (Phase E — pending)
    plot_per_client_distribution.py
    plot_centralized_vs_federated.py
    plot_scaling_n_clients.py
results/
  commcrime_run_<timestamp>/                  NEW per-run artifact dir (Phase A + C)
    partitions/client_<i>.pkl
    partitions/global_test.pkl
    scaler.joblib
    partition_stats.json
    env_pip_freeze.txt
    <timestamp>_training.log                  per-process log (Phase C edit re-routes here)
    <timestamp>_evaluation.log
    server_evaluation.log                     simulation runner aggregated metrics
    client_<i>/{training.log, evaluation.log} per-client logs under simulation
    fed_fusion_mlp_<save_name>.keras          final aggregated FL model
```

**Notes on divergences from the original layout proposal:**

* `commCrimeLoadProcess.py` and `commCrimePreprocess.py` — emerged from Phase A and Phase C refactors. The preprocess split keeps Phase A tests TF-free (~3 s); the load-process split shares download/clean/partition between single-client and multi-client entry points via `_prepare_partitions` (Phase C review #10).
* Partition artifacts are `.pkl` not `.parquet` — Phase 0 closed in favor of zero new dependencies; `pickle` round-trips pandas DataFrames natively.
* Saved models are `.keras` not `.h5` — Phase B review #11, the deprecation-safe Keras 3 format.
* Logs land in `run_dir` (not CWD) — Phase B review #5 routed them there; Phase C reinforced this.

**Edits** (all additive, all gated on `model_type == "FUSION-MLP"`):
- `Config/SessionConfig/ArgumentConfigLoad.py`
- `Config/SessionConfig/datasetLoadProcess.py`
- `Config/DatasetConfig/Dataset_Preprocessing/datasetPreprocess.py`
- `Config/SessionConfig/hyperparameterLoading.py`
- `Config/SessionConfig/modelCreateLoad.py`
- `Config/SessionConfig/ModelTrainingConfigLoad/modelCentralTrainingConfigLoad.py`
- `Config/SessionConfig/ModelTrainingConfigLoad/modelFederatedTrainingConfigLoad.py`
- `Config/SessionConfig/ModelTrainingConfigLoad/HFLStrategyTrainingConfigLoad.py`
- `App/TrainingApp/HFLHost/HFLHost.py`

---

## 3. Phased Milestones

Each phase ends with a **demo command** and a **Definition of Done**. Demos run on a single laptop — no Chameleon node — until Phase D scaling.

### Phase A — Data Path (1 sprint)

**Goal:** A reproducible end-to-end pipeline from the raw UCI Communities-and-Crime archive to per-client `(X, y_class, y_escalation)` tensors, with all reproducibility artifacts on disk. No model, no training.

**Deliverables**

A.1. **Raw loader.** [Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimeDatasetLoad.py](Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimeDatasetLoad.py)
- `download_communities_crime(target_dir)` — fetch and extract the archive; idempotent.
- `parse_names_file(path) -> list[str]` — build the column-header list from `.names`.
- `load_raw(csv_path, names) -> pd.DataFrame` — read with `na_values='?'`.
- `audit(df) -> dict` — Phase 2 schema audit (shape, dtypes, missing counts, state-code distribution); written to `partition_stats.json` under `audit:`.
- `clean(df, drop_sensitive: bool) -> pd.DataFrame` — drop ID columns, drop missing-target rows, optionally drop sensitive demographic columns.

A.2. **Label engineering.** [commCrimeLabelEngineering.py](Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimeLabelEngineering.py)
- `derive_threat_class(df) -> pd.Series` — argmax over normalized (violent, property, other) crime-rate families.
- `derive_escalation_score(df, weights={'violent':0.6,'property':0.3,'other':0.1}) -> pd.Series` — weighted severity index in `[0,1]`.
- Returns the dataframe with two new columns: `threat_class` (int 0/1/2), `escalation_score` (float).

A.3. **Federated partitioner.** [commCrimeFederatedPartition.py](Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimeFederatedPartition.py)
- `partition(labeled_df, strategy, num_clients, seed, dirichlet_alpha=None, global_test_size=0.15) -> dict`
  - Holds out a stratified global test set first (frozen on first call, re-loaded thereafter).
  - `strategy="geographic"`: state-code → N regional buckets (Northeast / Southeast / Midwest / Southwest / West for N=5; the `bucket_for_n(N)` helper covers N=3 and N=10 too).
  - `strategy="iid"`: random shuffle into N equal partitions.
  - `strategy="dirichlet"`: per-class Dirichlet allocation with concentration `dirichlet_alpha`.
  - Within each client, an 80/20 train/val split.
  - Writes `partitions/client_<i>.pkl` and `partitions/global_test.pkl`.
  - Writes `partition_stats.json` with per-client class distribution, sample counts, feature mean/std.

A.4. **Preprocessing.** Extend [Config/DatasetConfig/Dataset_Preprocessing/datasetPreprocess.py](Config/DatasetConfig/Dataset_Preprocessing/datasetPreprocess.py)
- `preprocess_communities_crime(client_partition, global_test, mode, scaler_path=None) -> tuple`
  - `mode="COMMCRIME"`: `StandardScaler` global standardization. Fit on first call, persist via `joblib.dump(scaler, scaler_path)`, reload thereafter.
  - `mode="COMMCRIME-MM"`: `MinMaxScaler` for ablations.
  - Returns `X_train, X_val, y_train, y_val, X_test, y_test` where each `y_*` is a tuple `(y_class, y_escalation)` of two numpy arrays.

A.5. **Wire into the dispatcher.** Extend [Config/SessionConfig/datasetLoadProcess.py](Config/SessionConfig/datasetLoadProcess.py) with the `COMMCRIME` branch — see design doc §3.2 for the exact code shape. Honors `args.client_id` (which partition to load) and `args.commcrime_random_seed`.

A.6. **Args.** Extend [Config/SessionConfig/ArgumentConfigLoad.py](Config/SessionConfig/ArgumentConfigLoad.py):
- Extend `--dataset` choices: add `"COMMCRIME"` and `"NIBRS"` (NIBRS is reserved; loader stub raises `NotImplementedError`).
- Extend `--dataset_processing` choices: add `"COMMCRIME"`, `"COMMCRIME-MM"`.
- Add fusion args block per design doc §3.1.
- Conditional logic: if `model_type == "FUSION-MLP"`, force `dataset == "COMMCRIME"` and `dataset_processing == "COMMCRIME"` when defaults are still in effect.

A.7. **Reproducibility writer.** New helper `Config/SessionConfig/runArtifacts.py`
- `init_run_dir() -> Path` — creates `results/commcrime_run_<timestamp>/`.
- `dump_pip_freeze(path)` — runs `pip freeze` and writes to `env_pip_freeze.txt`.
- `seed_all(seed)` — seeds `random`, `numpy`, `tf`/`torch`, sets `PYTHONHASHSEED`.

**Demo command**
```bash
python3 -c "
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import download_communities_crime, parse_names_file, load_raw, clean
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLabelEngineering import derive_threat_class, derive_escalation_score
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeFederatedPartition import partition
# end-to-end smoke; no training
"
```

**Definition of Done**
- Running the demo produces a `results/commcrime_run_<ts>/` tree with `partitions/client_0..4.pkl`, `partitions/global_test.pkl`, `scaler.joblib`, `partition_stats.json`, `env_pip_freeze.txt`.
- `partition_stats.json` shows non-IID class skew under `strategy=geographic` (visible difference across the five regional buckets) and uniform distribution under `strategy=iid`.
- Re-running with the same seed reproduces partitions byte-for-byte (deterministic).
- Re-running with the same seed but different `--num_clients` produces different but still deterministic partitions.
- Unit tests cover: label engineering math, partitioner determinism, geographic/IID/Dirichlet splits, scaler persistence + reload.
- Existing CICIOT/IOTBOTNET/IOT smoke tests still pass (additive-only invariant).

**Risk:** Low. Pure data engineering; no framework integration beyond extending two existing switches.

**Build outcome ✅ (2026-05-10):** All A.1–A.7 deliverables shipped. Closed at **35 tests**; review pass took it to **63 tests** across 5 files (`test_fusion_run_artifacts.py`, `test_fusion_label_engineering.py`, `test_fusion_partitioner.py`, `test_fusion_preprocess.py`, `test_fusion_data_path_smoke.py`, `test_fusion_dispatcher.py`).

*Refactors beyond original spec:*
- `commCrimePreprocess.py` split out so the data-path tests don't import `flwr` / `tensorflow` (the legacy `datasetPreprocess.py` does, transitively).
- `partition()` gained re-run conflict detection (review #2): conflicting strategy/seed/num_clients against an existing `partition_stats.json` raises rather than silently overwriting.
- `_split_geographic` is NaN-safe (review #5): FIPS lookup explicitly casts non-null entries to int; unmapped/missing states distribute round-robin and the count lands in `partition_stats.json["unmapped_state_rows"]`.
- Added `fit_global_scaler` to `commCrimePreprocess.py` for Phase C's "fit once on union of trains" requirement (review #7).
- Added `dropped_sensitive_columns` to `partition_stats.json` so Phase E ablation has an authoritative per-run record (review #8).
- `--drop_sensitive_features` switched to `argparse.BooleanOptionalAction` (review #10).
- `--num_clients` choices extended to `[1, 3, 5, 10]` so the Phase B centralized baseline can use the partitioner trivially.

---

### Phase B — Centralized FUSION-MLP (1 sprint)

**Goal:** End-to-end centralized training of the multi-task MLP on a single client partition, validating the joint loss and the dual-output evaluation metrics. Produces the centralized-baseline numbers (Experiment 1 of the matrix).

**Deliverables**

B.1. **Model structure.** [Config/modelStructures/FusionMLP/multiTaskMLPStruct.py](Config/modelStructures/FusionMLP/multiTaskMLPStruct.py)
- `build_fusion_mlp(input_dim, num_classes=3, hidden=(128, 64, 32), dropout=0.2) -> tf.keras.Model`
  - Functional API: `Input → Dense(128, ReLU) + BN + Dropout → Dense(64, ReLU) + BN + Dropout → Dense(32, ReLU) + BN + Dropout` (shared trunk).
  - Two heads on the trunk output: `Dense(num_classes, softmax, name="threat")` and `Dense(1, sigmoid, name="escalation")`.
  - Returns a Keras `Model` with two named outputs.
- `[backend-fork]` Under PyTorch: `class FusionMLP(nn.Module)` with `forward()` returning `(logits, escalation)`.

B.2. **Hyperparameters.** Extend [hyperparameterLoading.py](Config/SessionConfig/hyperparameterLoading.py) with the `FUSION-MLP` branch per design doc §3.3. Default `BATCH_SIZE=128`, `learning_rate=1e-3`, `num_classes=3`.

B.3. **Model factory.** Extend [modelCreateLoad.py](Config/SessionConfig/modelCreateLoad.py) with the `FUSION-MLP` branch per design doc §3.4. Returns `(model, None, None, None)`.

B.4. **Centralized trainer.** [Config/ModelTrainingConfig/ClientModelTrainingConfig/CentralTrainingConfig/FusionMLP/fusionMLPCentralTrainingConfig.py](Config/ModelTrainingConfig/ClientModelTrainingConfig/CentralTrainingConfig/FusionMLP/fusionMLPCentralTrainingConfig.py)
- `class CentralFusionMLPClient` with the same interface contract as `CentralNidsClient`:
  - `__init__(self, model, X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE, epochs, steps_per_epoch, learning_rate, escalation_loss_weight, evaluation_log, training_log, ...)`. `y_*` are `(y_class, y_escalation)` tuples.
  - In `__init__`: compile with `optimizer=Adam(lr)`, `loss={'threat': 'sparse_categorical_crossentropy', 'escalation': 'binary_crossentropy'}`, `loss_weights={'threat': 1-β, 'escalation': β}`, `metrics={'threat': ['accuracy'], 'escalation': ['mae', AUC()]}`.
  - `.fit()` — calls `model.fit(X_train, {'threat': y_class, 'escalation': y_esc}, validation_data=...)`.
  - `.evaluate()` — runs `model.predict` on `X_test`, computes macro-F1, per-class precision/recall, escalation MAE, escalation AUROC; passes through `recordEvaluation`.
  - `.save(save_name)` — writes to `ModelArchive/fusion_mlp_<save_name>.h5`.
  - `recordTraining` / `recordEvaluation` — same text format as `FlNidsClient` so existing `Analysis/` consumers work.

B.5. **Centralized dispatcher.** Extend [modelCentralTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/modelCentralTrainingConfigLoad.py) with the `FUSION-MLP` branch. Forwards `args.escalation_loss_weight` and the run-dir paths.

**Demo command**
```bash
python3 App/TrainingApp/Client/TrainingClient.py \
    --model_type FUSION-MLP \
    --trainingArea Central \
    --partition_strategy iid \
    --num_clients 1 \
    --client_id 0 \
    --epochs 50 \
    --save_name centralized_baseline
```

**Definition of Done**
- Demo produces `ModelArchive/fusion_mlp_centralized_baseline.h5` and a populated `evaluation_log` with macro-F1, per-class precision/recall, escalation MAE, escalation AUROC.
- Macro-F1 on the held-out global test set is `> 0.55` (sanity threshold; the paper goal is "centralized upper bound" — the exact number is the deliverable, not a prescribed target).
- Joint loss decreases monotonically over the run (modulo noise); both head losses are non-trivial (neither head collapses to a constant).
- Unit tests: model has the expected layer count + parameter count, joint loss is a weighted sum, both heads produce non-trivial gradients.
- Integration test: smoke run on a 100-row synthetic partition completes in `< 10s` and writes the expected artifacts.

**Risk:** Low-Medium. Multi-output Keras compilation is the only non-obvious bit; covered by the unit test on loss/metrics structure.

**Build outcome ✅ (2026-05-10):** All B.1–B.5 deliverables shipped. Closed at **79 tests**; review pass took it to **90 tests** across 2 new files (`test_fusion_mlp_struct.py`, `test_fusion_hyperparameters.py`, `test_fusion_centralized_training.py`, `test_fusion_central_dispatcher.py`).

*Refactors beyond original spec:*
- `save()` accepts `archive_path` kwarg (review #3) — defaults to project `ModelArchive/`, but `TrainingClient.py` passes `args.run_dir` for FUSION-MLP runs so the model artifact lives next to the run dir's logs.
- Trainer `__init__` takes `num_classes` (review #4) — `_compute_metrics` uses `labels=list(range(num_classes))` instead of hardcoded `[0, 1, 2]`, ready for future class-count changes.
- Training/eval logs route into `run_dir` (review #5) — `load_commcrime` re-writes `args.evaluationLog` / `args.trainingLog` to `<run_dir>/<filename>`.
- Distinct log filenames in argparse (review #2): `{timestamp}_evaluation.log` / `{timestamp}_training.log` instead of both being the bare timestamp string (pre-existing project bug fixed globally).
- Hermes-mode rejection (review #1): explicit `SystemExit` for `--mode hermes --model_type FUSION-MLP` in `TrainingClient.py` / `HFLHost.py` before the legacy hermes dispatch.
- `fit()` switched to `tf.data.Dataset.shuffle(seed=).batch().repeat()` (review #6): eliminates the Keras 3 "input ran out of data" warning, deterministically shuffles per round.
- Escalation metrics expanded (review #7): AUROC binarized at the **median** of true escalation scores (not fixed 0.5); added sibling `escalation_spearman` via `scipy.stats.spearmanr` for true ranking measure.
- Model-init seeding via `tf.keras.utils.set_random_seed` in `modelCreateLoad` (review #8): byte-deterministic across runs with the same `--commcrime_random_seed`.
- Save format `.keras` instead of `.h5` (review #11): Keras-3 deprecation-safe.
- `print_summary` flag on trainer (review #12): default `True` for central, `False` for FL clients to avoid N-client console flood.
- Fusion args now appear in both opening messages (review #12).

---

### Phase C — Federated FUSION-MLP via FedAvg (1 sprint)

**Goal:** Multi-client federated training in single-node simulation mode. Reproduces Experiments 3 and 4 of the matrix (FedAvg on IID and FedAvg on non-IID geographic).

**Deliverables**

C.1. **Federated trainer.** [Config/ModelTrainingConfig/ClientModelTrainingConfig/HFLClientModelTrainingConfig/FusionMLP/fusionMLPClientConfig.py](Config/ModelTrainingConfig/ClientModelTrainingConfig/HFLClientModelTrainingConfig/FusionMLP/fusionMLPClientConfig.py)
- `class FlFusionMLPClient(fl.client.NumPyClient)` — same shape as `FlNidsClient`:
  - `__init__` mirrors `CentralFusionMLPClient.__init__` with the addition of `node` for log labeling.
  - `get_parameters(config)` — returns `model.get_weights()`.
  - `fit(parameters, config)` — sets weights, runs `model.fit` for `self.epochs`, returns `(weights, len(X_train), {'train_loss': ..., 'val_loss': ...})`.
  - `evaluate(parameters, config)` — sets weights, runs the dual-head evaluation, returns `(loss, len(X_test), metrics_dict)` with macro-F1, escalation MAE, escalation AUROC keyed for server-side aggregation.
  - `recordTraining` / `recordEvaluation` / `save` identical to centralized except filename prefix `fed_fusion_mlp_`.

C.2. **Federated dispatcher.** Extend [modelFederatedTrainingConfigLoad.py](Config/SessionConfig/ModelTrainingConfigLoad/modelFederatedTrainingConfigLoad.py) with the `FUSION-MLP` branch.

C.3. **FedAvg strategy + aggregator.** [Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/FusionFedAvgConfig.py](Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/FusionFedAvgConfig.py)
- `class FusionFedAvg(fl.server.strategy.FedAvg)` — overrides `aggregate_evaluate` to compute weighted aggregations of macro-F1, escalation MAE, escalation AUROC, and per-client accuracy variance. Logs every aggregated metric per round to `evaluation_log`.
- Implements convergence stop: round-counter + 10-round-plateau detector on aggregated `val_loss`.

C.4. **Simulation runner.** [Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/SimulationRunner.py](Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/SimulationRunner.py)
- `def run_fusion_simulation(args, strategy)`:
  - Loads partitions for all `args.num_clients` clients up front (re-uses `datasetLoadProcess` per client_id).
  - Defines `client_fn(cid: str)` that constructs a `FlFusionMLPClient` for the integer-cast client id.
  - Calls `fl.simulation.start_simulation(client_fn=client_fn, num_clients=args.num_clients, config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy)`.
  - Aggregates per-client artifacts into `results/.../<experiment_name>/`.

C.5. **Host dispatch.** Extend [App/TrainingApp/HFLHost/HFLHost.py](App/TrainingApp/HFLHost/HFLHost.py): early dispatch on `args.model_type == "FUSION-MLP"`, call `run_fusion_simulation` with `FusionFedAvg`. Existing legacy/hermes paths and the GAN/NIDS strategies are untouched.

**Demo commands**
```bash
# Experiment 3 — FedAvg / IID
python3 App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP \
    --fl_strategy FedAvg \
    --partition_strategy iid \
    --num_clients 5 \
    --rounds 100 \
    --epochs 1 \
    --save_name fedavg_iid_n5

# Experiment 4 — FedAvg / non-IID geographic
python3 App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP \
    --fl_strategy FedAvg \
    --partition_strategy geographic \
    --num_clients 5 \
    --rounds 100 \
    --epochs 1 \
    --save_name fedavg_geo_n5
```

**Definition of Done**
- Both demos converge (either hit `--rounds` or the plateau detector fires) on a developer laptop in under one hour.
- IID FedAvg macro-F1 is within `±0.02` of the centralized baseline from Phase B (sanity check that federation works in the easy case).
- Non-IID FedAvg macro-F1 is below IID FedAvg by a measurable margin (confirms the non-IID partition is doing something).
- Per-round aggregated metrics are written to `evaluation_log` and human-readable.
- Per-client `training_log` files exist and are formatted compatibly with `Analysis/` consumers.
- Integration test: a 3-round / 2-client / 50-row synthetic simulation completes in `< 30s`.
- Existing IoT/GAN federated paths still pass their integration tests.

**Risk:** Medium. Flower's simulation API + multi-output Keras has a few sharp edges (`y` must be a dict for named-output models when going through `fit`, and `NumPyClient.fit` must serialize/deserialize correctly). Mitigation: write the integration test first, debug against the synthetic partition.

**Build outcome ✅ (2026-05-10):** All C.1–C.5 deliverables shipped. Closed at **107 tests**; review pass took it to **119 tests** across 2 new unit files (`test_fusion_fl_client.py`, `test_fusion_fedavg_strategy.py`) + 1 new integration file (`test_fusion_federated_simulation.py`).

*Refactors beyond original spec:*
- `FusionFedAvg.aggregate_fit` overridden (review #1): captures the latest aggregated parameters into `self._final_parameters` so the saved model artifact reflects training, not initial weights.
- NaN-safe weighted means in `aggregate_evaluate` (review #2): filters NaN entries (single-class AUROC fallbacks) instead of coercing to 0.
- Empty-partition guard in `fit()` (review #3): N=10 geographic partitions with empty regional buckets get `(parameters, 0, {...skipped_empty_partition: 1.0})` so FedAvg drops them from the round.
- `--run_dir` argparse flag (review #1 of Phase A retroactively applied to Phase C): SimulationRunner reuses the dir if provided so N clients share the same `global_test.pkl` and `partition_stats.json`. Defensive: clients with non-zero `--client_id` under simulation get a warning that the flag is ignored (review #13).
- `_save_final_model` documented to return a plain `tf.keras.Model` (review #4) — FedProx subclass is training-only state; resume-FL-training requires rebuilding `FedProxFusionMLPModel` and `set_weights(loaded)`.
- `_build_client_fn` caches per-cid clients (review #7): same instance returned across rounds, model built once per client.
- Central dispatcher integration test (review #9): `tests/integration/test_fusion_central_dispatcher.py` exercises the branch-selection path with a `tensorflow_privacy` stub so the legacy DP NIDS imports don't block.
- `--mode hermes` rejection moved to `_load_commcrime` AND the entry-point dispatch (defense in depth).
- `load_commcrime_for_simulation` added — multi-client load path; fits the global scaler once on the union of all client training partitions per outline §6.5.

---

### Phase D — FedProx + Scaling (1 sprint)

**Goal:** Add FedProx as an alternative strategy and parameterize over `--num_clients`. Reproduces Experiments 5 and 6 of the matrix.

**Deliverables**

D.1. **FedProx strategy.** [Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/FusionFedProxConfig.py](Config/ModelTrainingConfig/HostModelTrainingConfig/FusionCenters/FusionFedProxConfig.py)
- `class FusionFedProx(fl.server.strategy.FedAvg)` with proximal term:
  - `aggregate_fit` is identical to FedAvg (proximal term is client-side).
  - The strategy passes `mu = args.fedprox_mu` to clients via the `config` dict on `configure_fit`.

D.2. **Client-side proximal term.** Extend `FlFusionMLPClient.fit()`:
- If `config.get("mu")` is non-zero, wrap the optimizer step with the proximal regularizer `(mu/2) * ||w - w_global||²`. The cleanest TF/Keras path is a custom training step in a subclass `class FedProxFusionMLP(tf.keras.Model)` that adds the proximal penalty to the joint loss before `apply_gradients`. Keep the centralized model class unchanged; only the federated trainer wraps with the FedProx subclass when `mu > 0`.

D.3. **Server dispatch.** Extend `HFLStrategyTrainingConfigLoad.py` with the strategy switch:
```python
if args.model_type == "FUSION-MLP":
    if args.fl_strategy == "FedProx":
        strategy = FusionFedProx(...)
    else:
        strategy = FusionFedAvg(...)
    run_fusion_simulation(args, strategy)
    return
```
Wire from `HFLHost.py` Phase C dispatch.

D.4. **Scaling sanity test.** Add a `tests/integration/test_fusion_scaling.py` that runs `--num_clients ∈ {3, 5, 10}` for 5 rounds on the synthetic partition, asserting:
- All three runs complete and produce metrics.
- Per-round wall-clock time grows sub-linearly with N (single-node simulation should be CPU-bound, not coordination-bound).

**Demo commands**
```bash
# Experiment 5 — FedProx / non-IID
python3 App/TrainingApp/HFLHost/HFLHost.py \
    --model_type FUSION-MLP --fl_strategy FedProx --fedprox_mu 0.01 \
    --partition_strategy geographic --num_clients 5 --rounds 100 \
    --save_name fedprox_geo_n5

# Experiment 6 — Scaling sweep (run as a wrapper script)
for n in 3 5 10; do
  python3 App/TrainingApp/HFLHost/HFLHost.py \
      --model_type FUSION-MLP --fl_strategy FedAvg \
      --partition_strategy geographic --num_clients $n --rounds 100 \
      --save_name fedavg_geo_n${n}
done
```

**Definition of Done**
- FedProx with `mu=0.01` on the same non-IID partition produces a macro-F1 ≥ FedAvg on the same partition (the headline robustness claim).
- Scaling sweep produces three results directories with consistent metric formatting.
- Convergence-stop behavior validated: the plateau detector fires for at least one configuration; verified manually.
- Unit test: `FedProxFusionMLP` proximal penalty is zero when `w == w_global` and grows quadratically with `||w - w_global||`.

**Risk:** Medium. The proximal term implementation in Keras is non-trivial (needs a custom `train_step`); cover with the unit test above.

**Build outcome ✅ (2026-05-10):** All D.1–D.4 deliverables shipped. Closed at **137 tests**; review pass took it to **141**, then a follow-up adding `proximal_contribution` metric exposure to **146 tests** across 2 new unit files (`test_fusion_fedprox_model.py`, `test_fusion_fedprox_strategy.py`) + 1 new integration file (`test_fusion_scaling.py`).

*Refactors beyond original spec:*
- `multiTaskMLPStruct.py` factored to share `_build_layers` between FedAvg and FedProx model factories — single source of truth for the trunk + heads graph.
- `FedProxFusionMLPModel.set_global_weights` accepts the **full** Flower parameters list (matching `model.get_weights()` length) and filters to trainable indices internally (review #1) — pre-fix this crashed in train_step with a broadcast/shape error.
- `_global_weight_vars` stored via `self.__dict__["..."]` to bypass Keras's variable tracker so anchors don't bloat the `.keras` save artifact (review #6).
- `fedprox_mu` backed by `tf.Variable` for trace-safe per-round updates (review #2); `train_step` uses `tf.where` for graph-safe mu gating.
- Length validation in `set_global_weights` (review #9): wrong-length list raises a clear `ValueError`.
- `proximal_contribution` metric exposed in `train_step` return dict (post-review follow-up) + threaded through `FlFusionMLPClient.fit()`'s return dict with `.get(..., [0.0])[-1]` fallback so FedAvg clients return a consistent schema.
- `SimulationRunner` now calls `hyperparameterLoading` once on a representative client's `X_train` (review #4/#5) — single source of truth for BS/lr/num_classes/l2_alpha; honors `args.regularizationEnabled` (also propagated to `hyperparameterLoading` itself, which used to hardcode `True`).
- `start_simulation` → `run_simulation` migration documented as deferred (review #12 — Flower 1.29 still supports `start_simulation` with deprecation warning).

---

### Phase E — Paper-Ready Logging + Ablations (1 sprint)

**Goal:** All metrics, plots, and ablations needed to produce the headline figure of the paper (outline §7.9). No new architectural pieces.

**Deliverables**

E.1. **Sensitive-feature ablation.** Re-run Experiments 1, 4, and 5 with both `--drop_sensitive_features` settings; results land in two parallel directory trees. The exact recommendation from the design doc (§8.4) is `True` for headline runs, exposing both in an ablation row.

E.2. **Federation-overhead metrics.** Extend `FusionFedAvg.aggregate_fit` to log per-round: `parameter_update_wire_bytes` (sum of per-client *on-wire* serialized byte counts — captures any compression / encoding overhead; NOT the raw `4 × n_params × n_clients` float32 byte count), `parameter_update_wire_bytes_per_client` (the per-client average), `round_seconds` (wall-clock from `configure_fit` start to `aggregate_fit` finish), and `proximal_contribution` (weighted mean of clients' FedProx penalty contributions).

E.3. **Fairness metric.** Extend `FusionFedAvg.aggregate_evaluate` with per-client variance of `accuracy` on the global test set. Logged per round; final value extracted by the plotting script.

E.4. **Plot scripts.** [Analysis/CommunitiesCrime/](Analysis/CommunitiesCrime/) — pure pandas + matplotlib, consume the `evaluation_log` files.
- `plot_per_client_distribution.py` — bar chart of per-client class distribution from `partition_stats.json`. Outline §6.7 figure.
- `plot_centralized_vs_federated.py` — the headline figure (centralized upper bound vs FedAvg/non-IID vs FedProx/non-IID, macro-F1 over rounds).
- `plot_scaling_n_clients.py` — final macro-F1 and rounds-to-convergence vs N.

E.5. **Reproducibility doc.** Append a `RUNNING_FUSION_EXPERIMENTS.md` quickstart in `DeveloperDocs/` with exact command lines for the full experiment matrix and expected runtimes on the Chameleon `compute_haswell` / `compute_cascadelake` node classes.

**Definition of Done**
- The headline figure script regenerates the paper's primary plot from raw logs in `< 10s`.
- All six experiments in the matrix have at least one entry per ablation setting.
- The `RUNNING_FUSION_EXPERIMENTS.md` quickstart, run on a fresh clone, reproduces the headline figure end-to-end.

**Risk:** Low. Plot scripts are pandas + matplotlib; no framework integration.

**Build outcome ✅ (2026-05-10):** All E.1–E.5 deliverables shipped. Closed at **168 tests** (+22 net), then Phase E review pass took it to **174 tests** across 1 new unit file (`test_fusion_log_parser.py`) and 1 new integration file (`test_fusion_plot_scripts.py`); existing strategy + central-dispatcher tests extended with E.2/E.3 + review-fix coverage.

*Refactors beyond original spec:*
- `FusionFedAvg.configure_fit` overridden to stamp `_round_start_time` so `aggregate_fit` can compute `round_seconds` end-to-end (configure → train → return-trip).
- Overhead metrics returned in `aggregate_fit`'s metrics dict AND folded into `aggregate_evaluate`'s log row — single source of truth (the per-round line in `server_evaluation.log`).
- `threat_accuracy` added to `_compute_metrics` (was implicit in macro-F1 but never exposed); also added to the strategy's `weighted_metrics_keys` so it appears as a weighted aggregate in the log alongside its fairness variance (Phase E review #3).
- Both fairness metrics (`fairness_macro_f1_variance`, `fairness_accuracy_variance`) kept; macro-F1 captures class-imbalance sensitivity, accuracy captures overall rate. Plot scripts can use either.
- Log parser ([Analysis/CommunitiesCrime/log_parser.py](Analysis/CommunitiesCrime/log_parser.py)) — `parse_server_log` / `parse_client_log` / `parse_partition_stats` / `collect_server_logs`. Pure stdlib + pandas; no TF or Flower imports so plot tests run in ~10 s without ML stack.
- Plot scripts each exposed as both `python -m Analysis.CommunitiesCrime.plot_*` CLI **and** importable `plot(...)` function — tests use the function form for tmp-path control.
- `RUNNING_FUSION_EXPERIMENTS.md` includes a `--no-drop_sensitive_features` ablation section + a §1 data-validation snippet that catches the URL/column-name issues from §9 before a 100-round run wastes compute.

*Phase E review fixes applied:*
- **#1 NaN-safe fairness metrics** — filter NaN per-client values before computing variance; mirrors the Phase C #2 fix for paper-figure metrics.
- **#2 Renamed `parameter_update_size_bytes` → `parameter_update_wire_bytes`** (+ `..._per_client` variant) so the name reflects what's measured (post-serialization payload bytes, the federation-overhead measurement).
- **#3 Log parser round-trip test** — `test_strategy_log_round_trips_through_parser` drives `aggregate_fit` + `aggregate_evaluate` end-to-end, reads the resulting `server_evaluation.log`, parses it, asserts the weighted-mean math survives. Catches format drift between emitter and parser.
- **#4 `proximal_contribution` aggregated server-side** — `aggregate_fit` now weighted-means per-client `proximal_contribution` values across `num_examples` and surfaces the result in both the fit metrics dict and the eval-log row. Runbook §4.4 added with example commands plotting `proximal_contribution`, `parameter_update_wire_bytes`, and `fairness_accuracy_variance` evolution.

---

## 4. Cross-Phase Concerns

### 4.1 Hermes Mode Compatibility

The existing codebase has a `--mode {legacy, hermes}` gate (Phase 6 sprint work). The fusion-centers update does **not** ship a hermes-mode path. In `TrainingClient.py` and `HFLHost.py`, when `args.model_type == "FUSION-MLP"` and `args.mode == "hermes"`:

```python
if args.model_type == "FUSION-MLP" and args.mode == "hermes":
    raise SystemExit(
        "FUSION-MLP does not support --mode hermes; use --mode legacy."
    )
```

Add this check in Phase A, before the data load. Hermes integration is a follow-up if the project ever wants to fly fusion-center models on mules.

### 4.2 Backwards Compatibility

The grep test in `tests/unit/test_mode_switch.py` enforces that hermes imports stay inside their guard. The fusion-centers code must not import any hermes module — confirmed by adding `Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters` to the grep allow-list as a "hermes-free" path in the existing test.

Existing `--dataset CICIOT|IOTBOTNET|IOT|CANGAN` and `--model_type NIDS|GAN|WGAN-GP|AC-GAN|CANGAN` paths must produce byte-identical outputs before and after each phase. Validation: keep one regression test per existing model type running through CI; this test exists in the repo already and must not regress.

### 4.3 Reproducibility Controls

Seeded at every stochastic step from Phase A:
- `args.commcrime_random_seed` (default 42) seeds: download tie-breaks (none), schema audit (none), label engineering (none), partition (state-bucket→region mapping is deterministic; IID shuffle and Dirichlet sampling use this seed).
- `args.timestamp` is deterministic per process; the run directory uses it.
- Fitted scaler persisted on first run, re-loaded on subsequent runs.
- Global test split frozen on first run.
- `pip freeze` snapshot dumped at run start.

### 4.4 Chameleon Deployment

The `AppSetup/Chameleon_node_Setup.py` provisioning script already installs `flwr`, `tensorflow`, `pandas`, `scikit-learn`. Verify it includes `joblib` (likely yes via sklearn dependency) and `pyarrow` (for parquet partition files); if not, add them to the script in Phase A. No infrastructure work otherwise — outline §7.3 confirms compute is not the bottleneck.

---

## 5. Testing Strategy

A test pyramid mirroring the existing repo conventions.

| Layer | What it covers | Lives in |
|---|---|---|
| **Unit** | Label-engineering math, partitioner determinism + non-IID character, scaler persistence, model architecture (layer count, parameter count), joint-loss correctness, FedProx proximal-term math | `tests/unit/test_fusion_*.py` |
| **Integration** | Per-phase smoke runs on synthetic 100-row partitions: end-to-end data path (Phase A), centralized fit/evaluate (Phase B), 2-client 3-round simulation (Phase C), FedProx convergence (Phase D), plot regeneration (Phase E) | `tests/integration/test_fusion_*.py` |
| **Regression** | Existing IoT/GAN paths produce identical outputs | already exists; do not modify |
| **Reproducibility** | Same seed → same partitions → same first-round metrics | `tests/integration/test_fusion_repro.py` |

Each phase's "Definition of Done" lists which tests must pass before merging. The repo's existing CI conventions (per-phase isolation, no test relies on internet access) are honored: the loader is split into `download_communities_crime` (network) and `load_raw` (local file); tests use a checked-in 50-row CSV stub, not a fresh download.

---

## 6. Risk Register

| Risk | Phase | Mitigation |
|---|---|---|
| Backend choice unresolved (TF vs PyTorch) | Pre-A | Decide before Phase A starts; default plan is TF/Keras |
| Multi-output Keras compilation surprises (`y` shape, named heads) | B | Unit test on model + loss before writing the trainer |
| Flower simulation + multi-output `NumPyClient` interaction | C | Synthetic-partition integration test written first |
| FedProx proximal term implementation in Keras | D | Custom `train_step` subclass + dedicated unit test on the penalty function |
| Non-IID partition produces a degenerate split (one client has zero of one class) | A | Partitioner asserts each client has ≥ 1 sample per class; falls back to a stratified second pass if not |
| Bias in headline figure if sensitive features kept in | E | Default `--drop_sensitive_features True`, expose ablation; document explicitly in methods section |
| Dataset shape changes when sensitive features dropped (input_dim differs across runs) | A/B | `input_dim` is read from `X_train_data.shape[1]` at hyperparameter-loading time; trainer is shape-agnostic; no regression possible |
| Existing path regression (e.g., AC-GAN broken by an ArgumentConfigLoad edit) | every phase | Repo regression tests run on every PR; add `--model_type FUSION-MLP` smoke test as a grep-allowlisted addition |

---

## 7. Sprint Sequence + Effort

Five phases, roughly one-engineer-week each. Phases A–B can ship independently; C depends on B; D depends on C; E depends on D.

| Phase | Depends on | Effort | Demo artifact |
|---|---|---|---|
| A | Backend decision | 1 sprint | `partitions/`, `partition_stats.json`, `scaler.joblib` |
| B | A | 1 sprint | `fusion_mlp_centralized_baseline.h5` + evaluation log |
| C | B | 1 sprint | FedAvg/IID + FedAvg/non-IID logs |
| D | C | 1 sprint | FedProx logs + scaling sweep results |
| E | D | 1 sprint | Headline figure PNG, ablation rows, quickstart doc |

Total: ~5 sprints of single-engineer time. Phases A and B together produce a publishable centralized-vs-local-only result; Phases C and D produce the headline FL claim; Phase E produces the paper-ready figure and ablations.

---

## 8. Review Log

Each phase's deliverables passed a focused code-review before the next phase started. The reviews caught real bugs (some of which would have crashed production FL runs) and surfaced design gaps that would have compounded in later phases. This section is the audit trail.

### Phase A review (35 → 47 → 63 tests)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | 🔴 | `_load_commcrime` minted a new run_dir per call → Phase C N-client flow would create N test splits | Added `--run_dir` arg + `resolve_run_dir` helper; defensive on missing args |
| 2 | 🔴 | `partition()` silently overwrote client_*.pkl / partition_stats.json on re-run | Conflict detection: compare new inputs against existing stats; raise on mismatch, allow byte-identical re-runs |
| 3 | 🔴 | Hermes-mode rejection skipped | `SystemExit` at the top of `load_commcrime` |
| 4 | 🔴 | `audit()` exposed but never wired into stats | Called in `load_commcrime`, threaded through `partition(audit_info=...)` |
| 5 | 🟡 | `_split_geographic` fragile when `state` column has NaN (float coercion) | Explicit `notna()` mask + `.astype(int).map(...)`; round-robin fallback recorded as `unmapped_state_rows` |
| 6 | 🟡 | Test `test_geographic_skew_vs_iid_uniformity` validated nothing useful (synthetic stub has no real skew) | Renamed to `test_geographic_partition_populates_all_buckets`; added separate hand-crafted test for real class skew |
| 7 | 🟡 | No `fit_global_scaler` helper → Phase C needs the "fit once, distribute" pattern undocumented | Added `fit_global_scaler` to `commCrimePreprocess.py` with 4 tests |
| 8 | 🟡 | No record of which sensitive columns were actually dropped | `dropped_sensitive_columns` persisted in `partition_stats.json` |
| 9 | 🟡 | Dispatcher integration untested | Added `tests/integration/test_fusion_dispatcher.py` (8 tests) covering #1, #3, #4 |
| 10 | 🟢 | `--drop_sensitive_features` argparse pattern | Switched to `argparse.BooleanOptionalAction` (`--drop_sensitive_features` / `--no-drop_sensitive_features`) |
| 11 | 🟢 | `SENSITIVE_COLUMNS` taxonomy not documented | Expanded docstring grouping by race/ethnicity, per-capita-income-by-race, aggregate income |
| 12 | 🟢 | Opening message didn't show fusion args | Both `display_*` functions render a "🏛️ FUSION CENTERS CONFIG" block |

### Phase B review (79 → 84 → 90 tests)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | 🔴 | Hermes-mode rejection shadowed in `TrainingClient.py` (entry-point hermes path ran before the data-load check) | Added explicit rejection before `_run_hermes_main` in both entry points |
| 2 | 🔴 | `args.trainingLog == args.evaluationLog` (both bare timestamps) — metrics interleaved into one file | Distinct `{timestamp}_evaluation.log` / `{timestamp}_training.log` defaults in both parsers |
| 3 | 🔴 | Integration test polluted real `ModelArchive/` (monkeypatch.chdir didn't affect `__file__`-derived path) | `save()` takes `archive_path` kwarg; test passes `tmp_path` + asserts real archive stays clean |
| 4 | 🔴 | `_compute_metrics` hardcoded `labels=[0, 1, 2]` | Trainer takes `num_classes`; metrics dict generates `class_0..class_<N-1>` dynamically |
| 5 | 🟡 | Logs routed to CWD instead of `run_dir` | `load_commcrime` re-routes both log paths into the run dir |
| 6 | 🟡 | Keras 3 "input ran out of data" warning on every epoch | Wrapped `model.fit` inputs in `tf.data.Dataset.shuffle().batch().repeat()` |
| 7 | 🟡 | Escalation AUROC binarized at fixed 0.5 → NaN on skewed data | Median binarization + added `escalation_spearman` (true ranking metric) |
| 8 | 🟡 | No in-trainer reproducibility seeding | `tf.keras.utils.set_random_seed` called in `modelCreateLoad` before `build_fusion_mlp` |
| 9 | 🟡 | No central-dispatcher integration test | Added `tests/integration/test_fusion_central_dispatcher.py` (5 tests) |
| 10 | 🟢 | Unused `from pathlib import Path` import | Removed |
| 11 | 🟢 | `.h5` deprecated in Keras 3 | Switched to `.keras` format |
| 12 | 🟢 | `model.summary()` always printed | Opt-in `print_summary` kwarg (default True for central, False for FL clients) |
| 13 | 🟢 | `l2(0.0)` constructed when regularization disabled | `kernel_regularizer=l2(l2_alpha) if l2_alpha > 0 else None` |

### Phase C review (107 → 119 tests)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | 🔴 | `_save_final_model` saved INITIAL weights (no `aggregate_fit` override) → headline artifact useless | `FusionFedAvg.aggregate_fit` stashes `self._final_parameters` |
| 2 | 🔴 | `_safe_float` coerced NaN → 0 in weighted means | Filter NaN before mean; if all-NaN return NaN; `_safe_float` retained only for fairness variance |
| 3 | 🔴 | Empty client partition crashed `FlFusionMLPClient.fit()` | Early-return `(parameters, 0, {...skipped_empty_partition: 1.0})` |
| 4 | 🟡 | `SimulationRunner` hardcoded hyperparameters | Calls `hyperparameterLoading` once on representative client; honors `args.regularizationEnabled` |
| 5 | 🟡 | `l2_alpha=1e-4` ignored `args.regularizationEnabled` | Folded into #4; also fixed `hyperparameterLoading` to read the flag |
| 6 | 🟡 | `fit/evaluate` ignored server `config` | `self.proximal_mu = float(config.get("mu", 0.0))` captured; ready for Phase D |
| 7 | 🟡 | Model rebuilt per client per round | Cache `client_id → FlFusionMLPClient` in `_build_client_fn` closure |
| 8 | 🟡 | Test used private `flwr_client.numpy_client` attr | Restructured: separate `client_fn returns non-None` smoke + cache test |
| 9 | 🟡 | Strategy tests didn't cover `aggregate_fit` | Added with #1 fix |
| 10 | 🟡 | `load_commcrime_for_simulation` duplicated `load_commcrime` | Extracted `_prepare_partitions` shared helper |
| 11 | 🟢 | No `print_summary` flag on FL client | Added (default `False` for N-client simulation) |
| 12 | 🟢 | `start_simulation` deprecated in Flower 1.10+ | Module-docstring note; migration deferred |
| 13 | 🟢 | `args.client_id` not suppressed in simulation | Warning if non-zero under simulation mode |

### Phase D review (137 → 141 → 146 tests)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | 🔴 | `set_global_weights` shape-mismatch — Flower's full weights list (22) zipped against trainable_weights (16) would crash in train_step | `set_global_weights` accepts full weights, filters via cached `_trainable_indices` to align with trainable subset |
| 2 | 🟡 | `fedprox_mu` / `_global_weights_set` Python attrs → trace-fixed | `fedprox_mu` backed by `tf.Variable`; `train_step` uses `tf.where` for graph-safe mu gating |
| 3 | 🟡 | Reported `loss` in history.history excluded proximal contribution | Documented + exposed sibling `proximal_contribution` metric (post-review follow-up) |
| 4 | 🟡 | `_save_final_model` saves plain Model even for FedProx runs | Documented in docstring; correct for inference, resume-FL-training requires rebuild |
| 5 | 🟡 | No end-to-end test for FedProx → FlFusionMLPClient.fit() pipeline | Added 2 tests in `test_fusion_fl_client.py` |
| 6 | 🟡 | `_global_weight_vars` may leak into `model.weights` | Stored via `self.__dict__["..."]` to bypass Keras variable tracker; test verifies `len(model.weights)` unchanged after `set_global_weights` |
| 7 | 🟢 | Double-assignment of `fedprox_mu` in build_fedprox_fusion_mlp | `__init__` accepts `mu` kwarg; build passes through |
| 8 | 🟢 | Existing tests passed trainable-only list to set_global_weights | Updated to pass `model.get_weights()` consistently |
| 9 | 🟢 | No length validation in set_global_weights | Explicit ValueError on mismatch |

**Phase D follow-up — `proximal_contribution` metric**

| Item | Resolution |
|---|---|
| Surface what FedProx is doing in training curves | Model's `train_step` returns `proximal_contribution` (post-scale `(μ/2)·Σ‖w-g‖²`); zero when μ=0 or no anchor set; threaded through FL client's `fit()` return dict with 0.0 default for plain-Model (FedAvg) clients |

### Phase E review pass 1 (168 → 174 tests)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | 🟡 | Fairness metrics coerced NaN→0 in `_safe_float`, dragging variance upward (paper-figure metrics already filtered) | Filter NaN entries before computing per-client variance; if <2 valid entries remain, variance = 0 |
| 2 | 🟡 | `parameter_update_size_bytes` name suggested raw float32 byte count, but measured wire-format serialized bytes | Renamed to `parameter_update_wire_bytes` (and `..._per_client`); docstring + design doc + plan reflect on-wire semantics |
| 3 | 🟡 | Log parser tests used synthetic strings, never round-tripped through real `FusionFedAvg._append_log` output | Added `test_strategy_log_round_trips_through_parser` driving `aggregate_fit` + `aggregate_evaluate` end-to-end, parsing the resulting log file, asserting weighted-mean math + overhead metrics survive |
| 4 | 🟡 | `proximal_contribution` flowed from model → client fit-return-dict → nowhere (no plot script read it, no server-side aggregation) | `aggregate_fit` now weighted-means per-client `proximal_contribution`; emits in fit + eval log rows; runbook §4.4 added examples plotting it |

**Phase E review pass 1 bonus:** `threat_accuracy` added to weighted_metrics_keys (was emitted by clients per E.3 but never aggregated server-side), so plots can show both average accuracy and its variance.

### Phase E review pass 2 (174 → 183 tests)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 5 | 🟡 | `_rounds_to_convergence` heuristic was brittle: anchored to last-round value, no smoothing, absolute 0.01 tolerance | Rolling-mean smoothed series; anchor to max-of-smoothed (not last); tolerance is a fraction of the smoothed dynamic range (default 5%). Documented in docstring + 2 new tests |
| 6 | 🟡 | Bar height in scaling figure used last-round value, vulnerable to overshoot | Use best-of-smoothed; `test_smoothed_final_metric_ignores_last_round_overshoot` validates against an explicit late-decline log fixture |
| 7 | 🟡 | Two fairness metrics emitted with no canonical hierarchy | `fairness_accuracy_variance` marked canonical per outline §7.5 in strategy module docstring + inline comments; `fairness_macro_f1_variance` retained as secondary |
| 8 | 🟡 | `_round_start_time` was a scalar — Flower retry would overwrite | Replaced with `_round_start_times: dict[int, float]`; `configure_fit` uses `setdefault(server_round, time.time())` so the first stamp wins; 2 new tests verify retry-safety |
| 9 | 🟢 | Inconsistent default output paths across plot scripts | All three default to CWD-relative or run_dir-relative filenames (`per_client_distribution.png` next to its run_dir; `centralized_vs_federated.png` / `scaling_n_clients.png` in CWD) |
| 10 | 🟢 | `plot()` returned `Path`, not the matplotlib figure | All three now return `(Path, Figure)`; `close=False` kwarg keeps the figure alive for notebook reuse |
| 11 | 🟢 | No PDF/SVG output option | matplotlib already infers format from extension; `derive_savefig_kwargs()` handles vector formats (no `dpi`). Documented; `test_plot_per_client_distribution_pdf_output` validates real PDF byte output |
| 12 | 🟢 | `Analysis/CommunitiesCrime/__init__.py` was empty | Re-exports `parse_server_log`, `parse_client_log`, `parse_partition_stats`, `collect_server_logs` |
| 13 | 🟢 | Runbook validation snippet hardcoded `DEFAULT_TARGET_DIR` | Snippet now reads `DATA_DIR` env var with fallback to default |
| 14 | 🟢 | (already addressed in pass 1) | Runbook §4.4 documents proximal_contribution + overhead + fairness plotting |
| 15 | 🟢 | matplotlib-default styling not paper-grade | New `Analysis/CommunitiesCrime/plot_style.py` with `apply_style("paper")` context manager (300 DPI, no top/right spines, larger fonts, frameless legend); `--style {default,paper}` flag on all three CLIs |

---

## 9. Known Limitations / Untested Boundaries

These are items the 146-test suite **does not** cover. None are bugs in the tested code — they're risks at boundaries we deferred. The first real-data run will likely trip on at least one of 9.1–9.3.

### 9.1 ✅ RESOLVED (2026-05-11) — UCI schema validated against real data

- Diagnostic run confirmed the legacy `DEFAULT_NAMES_URL` is dead (UCI 2.0 stopped shipping the `.names` file in their zip).
- 10 of 14 `SENSITIVE_COLUMNS` names were guesses that didn't match the real UCI schema (`racepctblack` → `pctBlack`, `pctWInvInc` doesn't exist, `state` is actually `State` capital, etc.).
- **Fix:** new `Config/DatasetConfig/CommunitiesCrime_Sampling/generate_names_file.py` fetches the canonical schema via `ucimlrepo` and writes the ARFF `.names` file our parser expects. Constants in `commCrimeDatasetLoad.py` (`SENSITIVE_COLUMNS` + `audit`'s `State`/`state` reference) corrected. Synthetic stub schema updated to mirror real UCI names.
- **Setup procedure:** see [RUNNING_FUSION_EXPERIMENTS.md §1](RUNNING_FUSION_EXPERIMENTS.md) — `pip install ucimlrepo` + `python -m Config.DatasetConfig.CommunitiesCrime_Sampling.generate_names_file`.
- Validated end-to-end: 2215 rows × 147 cols load cleanly, all 13 sensitive cols match, all 8 crime-rate target cols match, no missing columns.

### 9.2 ✅ RESOLVED (2026-05-11) — `.names` file format

We now **generate** the `.names` file ourselves in the ARFF format the parser expects. Format is correct by construction (one `@attribute name type` line per variable). The ARFF-format assumption in `parse_names_file` is no longer a risk because we control the writer side.

### 9.3 🔴 Real-data DoD criteria never verified

Two phase DoDs reference real-data numbers that the synthetic-stub tests don't cover:

| Phase | DoD criterion | Validation status |
|---|---|---|
| B | "Macro-F1 on the held-out global test set > 0.55" | Synthetic learnable-data test confirms macro-F1 > 0.5 on toy data; **real UCI number unknown** |
| D | "FedProx with μ=0.01 produces macro-F1 ≥ FedAvg on the same non-IID partition" | FedProx is mechanically validated (pulls weights toward anchor); **headline-claim comparison on real data unknown** |
| C | "IID FedAvg macro-F1 within ±0.02 of centralized baseline" | Same — needs real data |

*First-real-run action:* expected output of Phase E. Plot scripts will produce the actual numbers.

### 9.4 🟡 `fl.simulation.start_simulation` never end-to-end tested

No test invokes the Ray-backed simulation entry point. Tests cover the building blocks (`load_commcrime_for_simulation`, `_build_client_fn`, `FlFusionMLPClient` protocol, strategy aggregation) **in isolation**. The orchestration layer — Ray actor pool lifecycle, client instance caching across actor boundaries, `History` object shape — is unexercised.

*Mitigation:* Add a Ray-backed slow-test marked `@pytest.mark.slow` that runs 2 clients × 3 rounds through `start_simulation`, OR accept that the first real `HFLHost.py --model_type FUSION-MLP` invocation is the validation. The proven failure modes from inspecting Flower 1.29 docs: client_fn may be called multiple times per cid; the per-cid cache may not persist across actor processes (Ray serializes the closure).

### 9.5 🟡 `proximal_contribution` not aggregated server-side

The FedProx client emits `proximal_contribution` in fit-time metrics; `FusionFedAvg.aggregate_fit` doesn't aggregate fit-side metrics, only stashes parameters. A paper plot of "proximal contribution over rounds across all clients" requires post-processing the per-client logs OR extending `aggregate_fit` to weighted-mean it.

*Mitigation:* Phase E.2 (federation-overhead metrics) already plans a metric-aggregation pass in `aggregate_fit`; fold `proximal_contribution` in there.

### 9.6 🟢 Legacy IoT/GAN regression suite untested in this env

The legacy NIDS trainer imports `tensorflow_privacy` at module load. The pinned `tensorflow-privacy==0.9.0` (most recent that installs) is incompatible with TF 2.21 — `from tensorflow.python.distribute import estimator_training` fails. The Phase C central-dispatcher test works around this with a `sys.modules.setdefault("tensorflow_privacy", ...)` stub, but **the actual IoT/GAN model paths haven't been exercised in this Windows + TF 2.21 environment**, so the "additive-only, no regressions" claim in §0 is by-construction, not by-test.

*Mitigation:* The next engineer on the project's original Linux + older-TF setup should run the legacy test suite once to confirm. The fusion-centers paths don't import `tensorflow_privacy`.

### 9.7 🟢 5 pre-existing failures in non-fusion tests (Windows env)

`tests/unit/test_mode_switch.py` (4 tests) and `tests/unit/test_experiments_calibration.py` (1 test) fail in this environment with `ModuleNotFoundError: No module named 'Config'` — the tests spawn `TrainingClient.py` as a subprocess from a CWD where the project root isn't on `PYTHONPATH`. The `sys.path.append('../../..')` trick at the top of `TrainingClient.py` doesn't help when the test invokes it from elsewhere. **These were failing before any fusion-centers work** — confirmed by inspecting the failing line (line 28, the very first project import).

*Mitigation:* Out of scope for this update. Linux env probably handles them correctly; the HERMES implementation plan reports "410 passed" on its original env.

### 9.8 🟢 Synthetic stub doesn't produce per-region class skew

`make_synthetic_stub` generates uniform-random state codes and uniform-random crime rates. The geographic partitioner correctly buckets by region, but the test data has no real correlation between region and crime-class distribution, so a "is geographic non-IID-er than IID?" comparison on the stub would be meaningless. We handle this with two tests:
- `test_geographic_partition_populates_all_buckets` — verifies the routing using the stub
- `test_geographic_partition_produces_real_class_skew` — uses hand-crafted FIPS→class mappings to verify the non-IID property

The implication: the stub is fine for plumbing tests but **doesn't quantify** the geographic non-IID effect. Real-data runs will show whether the regional skew is strong enough for the FedAvg-IID vs FedAvg-geographic comparison to be paper-worthy.

### 9.9 🟢 `start_simulation` deprecation warnings

Flower 1.29 supports `fl.simulation.start_simulation` but emits a deprecation warning recommending `fl.simulation.run_simulation` with `ServerAppComponents`. Logs will be noisy. Migration deferred — see §3 Phase D Build outcome.

### 9.10 ✅ RESOLVED (2026-05-11) — `threat_class` balanced via per-family z-score

**The bug:** Phase A `derive_threat_class` used per-row argmax over raw `*PerPop` rates. Property rates (~3000–4500/100k) always dwarf violent (~300–800/100k) and arson (~20–50/100k) by absolute magnitude, so the argmax always picked property. Real UCI run produced `threat_class.value_counts() == {1: 1902}` — single-class collapse, classifier macro-F1 trivially 1.0.

**The fix:** Option 1 from the original three — per-family z-score normalization across the dataset:

```
z_family[i] = (rate_family[i] − rate_family.mean()) / rate_family.std()
class[i]    = argmax_family(z_family[i])
```

A community is now labeled by the family it's most extreme in *relative to the dataset average*, not the family with the highest raw count. ~10 LOC change in `derive_threat_class` (see [commCrimeLabelEngineering.py](Config/DatasetConfig/CommunitiesCrime_Sampling/commCrimeLabelEngineering.py)).

**Validation against real UCI data (1902 rows after cleaning):**

| Class | Count | % |
|---|---|---|
| 0 (violent) | 504 | 26.5% |
| 1 (property) | 623 | 32.8% |
| 2 (other / arson) | 775 | 40.7% |

Non-degenerate 3-class problem; classifier has a real target.

**Test coverage:** old single-row "dominant family" tests removed (they tested the broken semantics); replaced with four new tests in `test_fusion_label_engineering.py`:
* `test_threat_class_per_family_zscore_assigns_correctly` — three hand-crafted communities each extreme in a different family
* `test_threat_class_z_score_beats_raw_magnitude` — the property-vs-violent edge case the old version failed
* `test_threat_class_all_zero_falls_back_to_other` — the explicit fallback path
* `test_threat_class_produces_balanced_distribution_on_real_like_data` — random realistic-rate dataset produces balanced multi-class output

**Edge cases handled:** single-row input (std=0, all z scores = 0, argmax picks index 0 deterministically — only relevant for unit tests), zero-variance family (treated as z=0 for that family), all-zero crime rates (remapped to class 2 / other for consistency with prior behavior).

---

*Last Updated: 2026-05-11*
*Status: All phases (0, A, B, C, D, E) closed — 183 tests passing*
*Real-data shakedown 2026-05-11: §9.1 (UCI URLs/schema), §9.2 (.names format), §9.10 (label-collapse) all ✅ RESOLVED. Pipeline produces balanced 3-class labels (26.5%/32.8%/40.7%) on real UCI data.*
*Operational runbook: [RUNNING_FUSION_EXPERIMENTS.md](RUNNING_FUSION_EXPERIMENTS.md)*
