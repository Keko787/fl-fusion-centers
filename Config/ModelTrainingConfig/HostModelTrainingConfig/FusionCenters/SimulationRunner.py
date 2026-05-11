"""Single-node Flower simulation runner for the fusion-centers FL update.

Phase C.4 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

The runner is the entry point ``HFLHost.py`` dispatches to when
``--model_type FUSION-MLP``. Pipeline:

  1. Load all client partitions + fit the global scaler
     (:func:`load_commcrime_for_simulation`).
  2. Call :func:`hyperparameterLoading` once on a representative client
     so BATCH_SIZE / learning_rate / l2_alpha / num_classes flow from
     one source of truth (Phase B's hyperparam loader, which honors
     ``args.regularizationEnabled``).
  3. Build the strategy chosen by ``--fl_strategy``
     (Phase C ships FedAvg; FedProx lands in Phase D).
  4. Define ``client_fn(cid)`` that constructs an
     :class:`FlFusionMLPClient` for the requested integer client id,
     pre-loaded with that client's preprocessed data. Per-client
     instances are cached so the model is built once per cid, not per
     (cid, round).
  5. Invoke ``fl.simulation.start_simulation`` with the round budget
     from ``--rounds``.
  6. Save the final aggregated model from
     :attr:`FusionFedAvg._final_parameters` (set by the strategy's
     ``aggregate_fit`` override).

Future-version note: Flower 1.10+ deprecated ``start_simulation`` in
favor of ``flwr.simulation.run_simulation`` with a
``ServerAppComponents`` config. The current call site still works on
Flower 1.29.x; migration is deferred until Flower drops the legacy API
or one of our experiments needs the new ``run_simulation`` features.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import flwr as fl
import numpy as np
import tensorflow as tf

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLoadProcess import (
    load_commcrime_for_simulation,
)
from Config.ModelTrainingConfig.ClientModelTrainingConfig.HFLClientModelTrainingConfig.FusionMLP.fusionMLPClientConfig import (
    FlFusionMLPClient,
)
from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.FusionFedAvgConfig import (
    FusionFedAvg,
)
from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.FusionFedProxConfig import (
    FusionFedProx,
)
from Config.SessionConfig.hyperparameterLoading import hyperparameterLoading
from Config.modelStructures.FusionMLP.multiTaskMLPStruct import (
    build_fedprox_fusion_mlp, build_fusion_mlp,
)


def _build_strategy(args, evaluation_log: str, initial_parameters):
    """Pick the strategy based on ``args.fl_strategy``.

    Phase C ships FedAvg; Phase D adds FedProx.
    """
    strategy_name = getattr(args, "fl_strategy", "FedAvg")
    common = dict(
        evaluation_log=evaluation_log,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_parameters,
    )
    if strategy_name == "FedAvg":
        return FusionFedAvg(**common)
    elif strategy_name == "FedProx":
        return FusionFedProx(
            fedprox_mu=getattr(args, "fedprox_mu", 0.01),
            **common,
        )
    else:
        raise ValueError(f"Unknown --fl_strategy: {strategy_name!r}")


def _build_client_fn(per_client: dict, run_dir: Path, args, *,
                       batch_size: int, learning_rate: float,
                       num_classes: int, l2_alpha: float,
                       use_fedprox: bool = False):
    """Return a ``client_fn`` suitable for ``fl.simulation.start_simulation``.

    Per-client instances are cached so the model + trainer are built
    once per cid (not per (cid, round)). Flower's Ray actor lifecycle
    may also pin a client to an actor across rounds; the cache makes
    repeat invocations cheap either way (Phase C review #7).

    When ``use_fedprox=True``, each client is built with the
    :class:`FedProxFusionMLPModel` subclass so the proximal-term
    machinery in ``train_step`` is available. The strategy still
    controls ``mu`` per round via the broadcast config.
    """
    X_test, y_test = per_client["global_test"]
    input_dim = X_test.shape[1]
    client_cache: dict[int, FlFusionMLPClient] = {}

    def client_fn(cid: str):
        client_id = int(cid)
        if client_id not in client_cache:
            slot = per_client[client_id]
            X_train, X_val = slot["X_train"], slot["X_val"]
            y_train, y_val = slot["y_train"], slot["y_val"]

            client_dir = run_dir / f"client_{client_id}"
            client_dir.mkdir(parents=True, exist_ok=True)

            if use_fedprox:
                model = build_fedprox_fusion_mlp(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    l2_alpha=l2_alpha,
                    mu=getattr(args, "fedprox_mu", 0.01),
                )
            else:
                model = build_fusion_mlp(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    l2_alpha=l2_alpha,
                )

            client_cache[client_id] = FlFusionMLPClient(
                model=model,
                x_train=X_train, x_val=X_val, x_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test,
                BATCH_SIZE=batch_size,
                epochs=args.epochs,
                steps_per_epoch=max(1, len(X_train) // batch_size),
                learning_rate=learning_rate,
                num_classes=num_classes,
                escalation_loss_weight=args.escalation_loss_weight,
                evaluation_log=str(client_dir / "evaluation.log"),
                training_log=str(client_dir / "training.log"),
                node=client_id,
                shuffle_seed=args.commcrime_random_seed + client_id,
                print_summary=False,
            )
        return client_cache[client_id].to_client()

    return client_fn


def _initial_weights(input_dim: int, num_classes: int,
                      l2_alpha: float, seed: int):
    """Build a throwaway model with the same arch to extract starting weights.

    Using deterministic initial parameters means all simulation runs
    with the same ``--commcrime_random_seed`` see the same initialization,
    so FedAvg results are reproducible round-by-round.
    """
    tf.keras.utils.set_random_seed(int(seed))
    seed_model = build_fusion_mlp(
        input_dim=input_dim, num_classes=num_classes, l2_alpha=l2_alpha,
    )
    weights = seed_model.get_weights()
    return fl.common.ndarrays_to_parameters(weights)


def run_fusion_simulation(args):
    """Drive the end-to-end Phase C single-node Flower simulation."""
    # ── 1. Data ─────────────────────────────────────────────────────────
    per_client = load_commcrime_for_simulation(args)
    run_dir = Path(per_client["run_dir"])
    X_test, _ = per_client["global_test"]

    # ── 2. Hyperparameters (single source of truth, Phase B loader) ────
    # Use client 0's training data as the representative sample for
    # input_dim / steps_per_epoch defaults. The simulation re-derives
    # per-client steps_per_epoch in the client factory.
    sample_X_train = per_client[0]["X_train"]
    (BATCH_SIZE, _noise_dim, _spe, input_dim, num_classes, _latent,
     _betas, learning_rate, l2_alpha, *_callback_slots) = hyperparameterLoading(
        args, sample_X_train,
    )
    # Honor args.regularizationEnabled at the runner level too (the
    # loader returns l2_alpha=1e-4 unconditionally for FUSION-MLP, but
    # the runner can still zero it out if the user asked).
    if not getattr(args, "regularizationEnabled", True):
        l2_alpha = 0.0

    # ── 3. Strategy ────────────────────────────────────────────────────
    server_log = run_dir / "server_evaluation.log"
    initial_parameters = _initial_weights(
        input_dim=input_dim, num_classes=num_classes, l2_alpha=l2_alpha,
        seed=args.commcrime_random_seed,
    )
    strategy = _build_strategy(args, str(server_log), initial_parameters)

    # ── 4. Client factory ──────────────────────────────────────────────
    client_fn = _build_client_fn(
        per_client, run_dir, args,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        num_classes=num_classes,
        l2_alpha=l2_alpha,
        use_fedprox=(args.fl_strategy == "FedProx"),
    )

    # ── 5. Run simulation ──────────────────────────────────────────────
    print(f"\n=== FusionFedAvg simulation: "
          f"num_clients={args.num_clients}, rounds={args.rounds}, "
          f"strategy={args.fl_strategy}, partition={args.partition_strategy} ===")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    # ── 6. Save final aggregated model ─────────────────────────────────
    final = _save_final_model(
        strategy, input_dim=input_dim, num_classes=num_classes,
        l2_alpha=l2_alpha, run_dir=run_dir, save_name=args.save_name,
    )
    print(f"=== Simulation complete. Final model: {final} ===")
    print(f"=== Aggregated history: {len(strategy.history)} rounds, "
          f"plateau_detected={strategy.plateau_detected} ===")

    return {
        "run_dir": str(run_dir),
        "history": history,
        "strategy_history": strategy.history,
        "final_model_path": final,
        "plateau_detected": strategy.plateau_detected,
    }


def _save_final_model(strategy, *, input_dim: int, num_classes: int,
                       l2_alpha: float, run_dir: Path,
                       save_name: str) -> str:
    """Reconstruct the trained model from the strategy's final parameters.

    Relies on :meth:`FusionFedAvg.aggregate_fit` having captured the
    latest aggregated parameters into ``strategy._final_parameters``
    (Phase C review #1). Falls back to ``initial_parameters`` only if
    aggregation never produced output (e.g. all rounds failed); a
    visible warning is printed in that case.

    Even for FedProx runs we save a **plain** ``tf.keras.Model`` (not
    the ``FedProxFusionMLPModel`` subclass). The proximal term is
    training-only state, so inference and evaluation work identically.
    To *resume FL training* from a saved FedProx artifact: load the
    plain model, build a fresh ``FedProxFusionMLPModel`` with the same
    architecture, and call ``set_weights(loaded.get_weights())`` —
    documented in :func:`load_fusion_mlp`.
    """
    final_params = strategy._final_parameters
    if final_params is None:
        print("⚠️  No aggregated parameters captured — saving INITIAL weights. "
              "This means every FL round produced no successful results.")
        final_params = strategy.initial_parameters

    weights = fl.common.parameters_to_ndarrays(final_params)
    model = build_fusion_mlp(
        input_dim=input_dim, num_classes=num_classes, l2_alpha=l2_alpha,
    )
    model.set_weights(weights)
    out_path = run_dir / f"fed_fusion_mlp_{save_name}.keras"
    model.save(out_path)
    return str(out_path)
