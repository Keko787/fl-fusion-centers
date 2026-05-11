"""Distributed FUSION-MLP Flower server (real multi-node federation).

Counterpart to :func:`SimulationRunner.run_fusion_simulation`. Where the
simulation runner spins up every client in-process via
``fl.simulation.start_simulation``, this runner binds a real Flower
gRPC server on the network. Each client runs ``TrainingClient.py
--trainingArea Federated --model_type FUSION-MLP --custom-host <ip>``
on its own machine and dials in. Selected by
``HFLHost.py --distributed``.

Per-client preprocessing fits its own scaler from the client's local
training partition (each :func:`load_commcrime` call writes a local
``scaler.joblib``). The cross-client global standardization the
simulation does is NOT performed in distributed mode — clients cannot
see each other's data without an out-of-band scaler distribution step.

Strategy, initial parameters, and final-model save behavior are shared
with the simulation runner via imported helpers.
"""
from __future__ import annotations
from pathlib import Path

import flwr as fl

from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeLoadProcess import (
    load_commcrime,
)
from Config.SessionConfig.hyperparameterLoading import hyperparameterLoading
from Config.ModelTrainingConfig.HostModelTrainingConfig.FusionCenters.SimulationRunner import (
    _build_strategy, _initial_weights, _save_final_model,
)


def run_fusion_distributed_server(args, server_address: str = "[::]:8080") -> dict:
    """Drive a real Flower server for FUSION-MLP across networked clients.

    Loads client-0's partition only to derive ``input_dim`` /
    ``num_classes`` / ``l2_alpha`` (needed for ``initial_parameters``
    and the final-model reconstruction). The actual training data is
    discarded — clients fetch and own their own partitions on their
    own machines.
    """
    # ── 1. Schema probe (cheapest available — reuses partition logic). ──
    args.client_id = 0
    X_train, *_ = load_commcrime(args)
    (BATCH_SIZE, _noise_dim, _spe, input_dim, num_classes, _latent,
     _betas, learning_rate, l2_alpha, *_) = hyperparameterLoading(
        args, X_train,
    )
    if not getattr(args, "regularizationEnabled", True):
        l2_alpha = 0.0

    run_dir = Path(args.run_dir)

    # ── 2. Strategy + seeded initial parameters. ───────────────────────
    server_log = run_dir / "server_evaluation.log"
    initial_parameters = _initial_weights(
        input_dim=input_dim, num_classes=num_classes, l2_alpha=l2_alpha,
        seed=args.commcrime_random_seed,
    )
    strategy = _build_strategy(args, str(server_log), initial_parameters)

    # ── 3. Bind. ───────────────────────────────────────────────────────
    print(
        f"\n=== FUSION-MLP distributed server on {server_address} "
        f"(strategy={args.fl_strategy}, rounds={args.rounds}, "
        f"min_clients={args.min_clients}) ===\n"
        f"Run each client with: \n"
        f"  python App/TrainingApp/Client/TrainingClient.py \\\n"
        f"    --model_type FUSION-MLP --trainingArea Federated \\\n"
        f"    --custom-host <THIS_HOST_IP> \\\n"
        f"    --partition_strategy {args.partition_strategy} "
        f"--num_clients {args.num_clients} --client_id <0..{args.num_clients-1}> \\\n"
        f"    --commcrime_random_seed {args.commcrime_random_seed} \\\n"
        f"    --epochs {args.epochs} --save_name <per-client-name>\n"
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # ── 4. Save final aggregated model. ────────────────────────────────
    final = _save_final_model(
        strategy, input_dim=input_dim, num_classes=num_classes,
        l2_alpha=l2_alpha, run_dir=run_dir, save_name=args.save_name,
    )
    print(f"=== Server done. Final model: {final} ===")
    return {"run_dir": str(run_dir), "final_model_path": final}
