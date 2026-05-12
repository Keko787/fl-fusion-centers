"""Dispatcher-level orchestration for the COMMCRIME data path.

Phase A.5 + Phase C of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

Lives in the COMMCRIME package (not in ``Config/SessionConfig/``) so the
Phase A integration test can import it without pulling in
``tensorflow`` / ``flwr`` through ``datasetLoadProcess``'s legacy
top-level imports.

Two public entry points:
  * :func:`load_commcrime` — Phase A.5; preprocesses one client's slice
    and returns the standard 6-tuple. Used by single-client centralized
    runs and real-multi-process FL.
  * :func:`load_commcrime_for_simulation` — Phase C; preprocesses all
    clients with a shared global scaler. Used by the single-node Flower
    simulation runner.

Both share the download → clean → label-engineer → partition pipeline
via the private :func:`_prepare_partitions` helper.
"""
from __future__ import annotations

from pathlib import Path

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    DEFAULT_TARGET_DIR, RAW_NAMES_FILENAME, SENSITIVE_COLUMNS,
    audit, clean, download_communities_crime, load_raw, parse_names_file,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeFederatedPartition import (
    partition,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLabelEngineering import (
    engineer_labels,
)
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimePreprocess import (
    fit_global_scaler, preprocess_communities_crime,
)
from Config.SessionConfig.runArtifacts import (
    dump_pip_freeze, resolve_run_dir, seed_all,
)


def _prepare_partitions(args) -> dict:
    """Shared download → clean → label-engineer → partition pipeline.

    Used by both :func:`load_commcrime` (single client) and
    :func:`load_commcrime_for_simulation` (all clients). Mutates ``args``
    with ``run_dir`` and re-routes ``evaluationLog`` / ``trainingLog``
    into the run directory.

    Returns ``{"partitions": <partition output>, "run_dir": Path,
                "audit_info": dict, "dropped_sensitive": list[str]}``.
    """
    # Hermes-mode rejection (impl plan §4.1).
    if getattr(args, "mode", None) == "hermes":
        raise SystemExit(
            "FUSION-MLP / COMMCRIME does not support --mode hermes; use --mode legacy."
        )

    seed_all(args.commcrime_random_seed)

    # ── 1. Resolve raw paths and ensure files on disk ──
    if getattr(args, "commcrime_path", None):
        data_path = Path(args.commcrime_path)
        names_path = data_path.with_name(RAW_NAMES_FILENAME)
        if not names_path.exists():
            names_path = data_path.parent / RAW_NAMES_FILENAME
    else:
        data_path, names_path = download_communities_crime(DEFAULT_TARGET_DIR)

    # ── 2. Load + audit + clean + label-engineer ──
    names = parse_names_file(names_path)
    raw_df = load_raw(data_path, names)
    audit_info = audit(raw_df)
    if args.drop_sensitive_features:
        dropped_sensitive = [c for c in SENSITIVE_COLUMNS if c in raw_df.columns]
    else:
        dropped_sensitive = []
    cleaned = clean(raw_df, drop_sensitive=args.drop_sensitive_features)
    labeled = engineer_labels(cleaned)

    # ── 3. Run-dir + reproducibility artifacts ──
    run_dir = resolve_run_dir(getattr(args, "run_dir", None),
                              timestamp=args.timestamp)
    dump_pip_freeze(run_dir / "env_pip_freeze.txt")

    # ── 4. Partition (conflict-detected; idempotent on same inputs) ──
    partitions = partition(
        labeled,
        strategy=args.partition_strategy,
        num_clients=args.num_clients,
        seed=args.commcrime_random_seed,
        run_dir=run_dir,
        dirichlet_alpha=(args.dirichlet_alpha
                         if args.partition_strategy == "dirichlet" else None),
        global_test_size=args.global_test_size,
        audit_info=audit_info,
        dropped_sensitive_columns=dropped_sensitive,
    )

    # ── 5. Mutate args so downstream trainers/loggers find the run dir ──
    args.run_dir = str(run_dir)
    if getattr(args, "evaluationLog", None):
        args.evaluationLog = str(run_dir / Path(args.evaluationLog).name)
    if getattr(args, "trainingLog", None):
        args.trainingLog = str(run_dir / Path(args.trainingLog).name)

    return {
        "partitions": partitions,
        "run_dir": run_dir,
        "audit_info": audit_info,
        "dropped_sensitive": dropped_sensitive,
    }


def load_commcrime(args):
    """Build the per-client COMMCRIME partition and return the standard 6-tuple.

    Phase A.5 entry. Pipeline runs through :func:`_prepare_partitions`
    and then preprocesses the single client identified by
    ``args.client_id``. Returns ``(X_train, X_val, y_train, y_val,
    X_test, y_test)`` where the ``y_*`` slots are
    ``(y_class, y_escalation)`` tuples.
    """
    prep = _prepare_partitions(args)
    partitions = prep["partitions"]
    run_dir = prep["run_dir"]

    client_id = int(args.client_id)
    if client_id not in partitions:
        raise ValueError(
            f"--client_id {client_id} out of range for --num_clients {args.num_clients}"
        )
    client_train = partitions[client_id]["train"]
    client_val = partitions[client_id]["val"]
    global_test = partitions["global_test"]

    return preprocess_communities_crime(
        client_train, client_val, global_test,
        mode=args.dataset_processing,
        scaler_path=str(run_dir / "scaler.joblib"),
    )


def load_commcrime_via_simulation(args):
    """Client-side loader that uses the same global scaler as the simulation.

    Counterpart to :func:`load_commcrime` for distributed FL. Where
    ``load_commcrime`` fits a scaler from the calling client's training
    partition alone, this function fits the scaler on the **union** of
    every client's training partition — exactly what
    :func:`load_commcrime_for_simulation` does for in-process runs —
    then returns only the ``--client_id`` slice. Used when
    ``TrainingClient.py --trainingArea Federated --global_scaler`` so
    distributed runs are bit-comparable to the simulation at the same
    ``--commcrime_random_seed``.

    Returns the standard ``(X_train, X_val, y_train, y_val, X_test,
    y_test)`` 6-tuple. ``y_train`` / ``y_val`` / ``y_test`` are
    ``(y_class, y_escalation)`` pairs, matching every other COMMCRIME
    loader.

    Every distributed client must have the full COMMCRIME raw archive
    on its local filesystem — the partition step reads every row to
    assign it to a client, regardless of which slice is kept here.
    """
    cid = int(args.client_id)
    if cid < 0 or cid >= int(args.num_clients):
        raise ValueError(
            f"--client_id {cid} out of range for --num_clients {args.num_clients}"
        )
    # Mask client_id around the underlying call to avoid the simulation
    # path's "client_id is ignored" warning (which is true for the
    # simulation but misleading here — we DO consume the client_id, just
    # at the extract step rather than the partition step).
    saved_cid = args.client_id
    args.client_id = 0
    try:
        sim = load_commcrime_for_simulation(args)
    finally:
        args.client_id = saved_cid
    slot = sim[cid]
    X_test, y_test = sim["global_test"]
    return (
        slot["X_train"], slot["X_val"],
        slot["y_train"], slot["y_val"],
        X_test, y_test,
    )


def load_commcrime_for_simulation(args) -> dict:
    """Phase C entry — load all clients' partitions in one pass.

    Returns a dict ``{client_id: {"X_train", "X_val", "y_train",
    "y_val"}, ..., "global_test": (X_test, y_test), "run_dir": str}``
    where:

      * Every client share has been preprocessed using the **same**
        scaler — fit once on the union of all client training
        partitions (outline §6.5 global standardization).
      * The global test set is the frozen Phase A split.
      * The run directory is materialized and returned so the
        simulation runner can park logs / models alongside it.

    Honors hermes-mode rejection via :func:`_prepare_partitions`.
    """
    # #13: simulation mode loads ALL clients; args.client_id is unused
    # here. Warn so a user passing --client_id 7 --num_clients 5 doesn't
    # silently end up with their flag ignored.
    if int(getattr(args, "client_id", 0) or 0) != 0:
        print(
            f"⚠️  --client_id={args.client_id} is ignored under simulation mode "
            f"(all {args.num_clients} clients are loaded together)."
        )

    prep = _prepare_partitions(args)
    partitions = prep["partitions"]
    run_dir = prep["run_dir"]

    # ── Fit the global scaler once on the union of all client trains ──
    client_trains = [partitions[cid]["train"] for cid in range(args.num_clients)]
    scaler_path = fit_global_scaler(
        client_trains, str(run_dir), mode=args.dataset_processing,
    )

    # ── Preprocess each client's slice with the same scaler ──
    out: dict = {"run_dir": str(run_dir)}
    for cid in range(args.num_clients):
        X_train, X_val, y_train, y_val, X_test, y_test = preprocess_communities_crime(
            partitions[cid]["train"],
            partitions[cid]["val"],
            partitions["global_test"],
            mode=args.dataset_processing,
            scaler_path=str(scaler_path),
        )
        out[cid] = {
            "X_train": X_train, "X_val": X_val,
            "y_train": y_train, "y_val": y_val,
        }
        out.setdefault("global_test", (X_test, y_test))

    return out
