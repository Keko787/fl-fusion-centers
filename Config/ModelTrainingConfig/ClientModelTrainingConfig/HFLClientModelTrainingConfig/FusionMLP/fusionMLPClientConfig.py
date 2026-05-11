"""Federated multi-task MLP client for the fusion-centers update.

Phase C.1 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

The class mirrors :class:`FlNidsClient` but operates on the
``(y_class, y_escalation)`` tuple labels produced by
``preprocess_communities_crime``. Joint loss, evaluation metrics, and
``tf.data.Dataset.repeat()`` wrapping are duplicated from
:class:`CentralFusionMLPClient` rather than inherited — the FL trainer
is a peer, not a subclass, to keep Flower's NumPyClient interface
clean and to avoid surprising counter / log side-effects from the
centralized client's ``fit()``.
"""
from __future__ import annotations

import os
import time

import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.FusionMLP.fusionMLPCentralTrainingConfig import (
    CentralFusionMLPClient,
)


class FlFusionMLPClient(fl.client.NumPyClient):
    """Flower NumPyClient for the fusion-centers multi-task MLP."""

    def __init__(self,
                 model,
                 x_train, x_val, x_test,
                 y_train, y_val, y_test,
                 BATCH_SIZE, epochs, steps_per_epoch,
                 learning_rate,
                 num_classes: int = 3,
                 escalation_loss_weight: float = 0.5,
                 evaluation_log: str = "fed_fusion_mlp_eval.log",
                 training_log: str = "fed_fusion_mlp_train.log",
                 node: int = 1,
                 model_name: str = "fed_fusion_mlp",
                 shuffle_seed: int = 0,
                 print_summary: bool = False):
        self.model = model

        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train_class, self.y_train_esc = y_train
        self.y_val_class, self.y_val_esc = y_val
        self.y_test_class, self.y_test_esc = y_test

        self.batch_size = BATCH_SIZE
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.num_classes = int(num_classes)
        self.escalation_loss_weight = float(escalation_loss_weight)

        self.evaluation_log = evaluation_log
        self.training_log = training_log
        self.node = int(node)
        self.model_name = model_name
        self.shuffle_seed = int(shuffle_seed)

        # counters
        self.round_count = 0
        self.evaluate_count = 0

        # --- compile with joint loss (same recipe as CentralFusionMLPClient) ---
        beta = self.escalation_loss_weight
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                "threat": "sparse_categorical_crossentropy",
                "escalation": "binary_crossentropy",
            },
            loss_weights={"threat": 1.0 - beta, "escalation": beta},
            metrics={
                "threat": ["accuracy"],
                "escalation": ["mae"],
            },
        )
        if print_summary:
            self.model.summary()

    # ───────────────────────────────────────────────────────────────────
    #   Flower NumPyClient contract
    # ───────────────────────────────────────────────────────────────────

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.round_count += 1
        start = time.time()

        # Apply server-broadcast weights for this round
        self.model.set_weights(parameters)

        # Phase D — FedProx proximal-term wiring. The strategy's
        # ``configure_fit`` broadcasts ``mu`` in the per-round config.
        # When the local model is a ``FedProxFusionMLPModel`` (has
        # ``set_global_weights``) and mu > 0, anchor the proximal term
        # against the just-received server weights so the train_step
        # adds ``(mu/2) * ||w - w_global||²`` to the loss. With FedAvg
        # the broadcast is absent (or mu == 0) and the term stays inert.
        self.proximal_mu = float(config.get("mu", 0.0)) if config else 0.0
        if hasattr(self.model, "set_global_weights"):
            self.model.set_global_weights(parameters)
            self.model.fedprox_mu = self.proximal_mu

        # #3 — Empty client partition: return immediately with zero
        # weight so FedAvg ignores this client this round. Possible
        # with N=10 geographic where small regions land empty after
        # the train/val split.
        if len(self.x_train) == 0:
            return (
                self.model.get_weights(),
                0,
                {"loss": float("nan"),
                 "val_loss": float("nan"),
                 "threat_loss": float("nan"),
                 "escalation_loss": float("nan"),
                 "val_threat_loss": float("nan"),
                 "val_escalation_loss": float("nan"),
                 "skipped_empty_partition": 1.0},
            )

        # Same tf.data.Dataset wrapping as the centralized trainer so
        # steps_per_epoch is honored without "input ran out of data"
        # warnings (Phase B issue #6).
        train_ds = (
            tf.data.Dataset.from_tensor_slices((
                self.x_train,
                {"threat": self.y_train_class,
                 "escalation": self.y_train_esc},
            ))
            .shuffle(buffer_size=len(self.x_train),
                     seed=self.shuffle_seed + self.round_count,
                     reshuffle_each_iteration=True)
            .batch(self.batch_size)
            .repeat()
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((
                self.x_val,
                {"threat": self.y_val_class,
                 "escalation": self.y_val_esc},
            ))
            .batch(self.batch_size)
        )

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=0,
        )

        elapsed = time.time() - start
        self._record_training(history, elapsed)

        return (
            self.model.get_weights(),
            len(self.x_train),
            {
                "loss": float(history.history["loss"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "threat_loss": float(history.history["threat_loss"][-1]),
                "escalation_loss": float(history.history["escalation_loss"][-1]),
                "val_threat_loss": float(history.history["val_threat_loss"][-1]),
                "val_escalation_loss": float(history.history["val_escalation_loss"][-1]),
                # FedProx-only metric: the post-scale ``(mu/2)·Σ||w-g||²``
                # the proximal term contributed at the last step. Plain
                # FedAvg models don't emit this key — fall back to 0 so
                # the metrics dict has a consistent schema across
                # strategies.
                "proximal_contribution": float(
                    history.history.get("proximal_contribution", [0.0])[-1]
                ),
            },
        )

    def evaluate(self, parameters, config):
        # ``config`` is unused for evaluation in Phase C; reserved for
        # Phase D / future strategies that may broadcast eval-specific
        # parameters (different metric set, custom thresholds, etc.).
        self.evaluate_count += 1
        start = time.time()

        self.model.set_weights(parameters)

        # #3 — Empty global test would be a bug elsewhere, but guard
        # symmetrically with fit() just in case.
        if len(self.x_test) == 0:
            return (float("nan"), 0,
                    {"skipped_empty_partition": 1.0})

        threat_pred, escalation_pred = self.model.predict(
            self.x_test, batch_size=self.batch_size, verbose=0,
        )
        eval_results = self.model.evaluate(
            self.x_test,
            {"threat": self.y_test_class,
             "escalation": self.y_test_esc},
            batch_size=self.batch_size, verbose=0, return_dict=True,
        )

        # Reuse the centralized client's static metrics helper.
        metrics = CentralFusionMLPClient._compute_metrics(
            self.y_test_class, threat_pred,
            self.y_test_esc, escalation_pred,
            num_classes=self.num_classes,
        )
        metrics["total_loss"] = float(eval_results.get("loss", 0.0))
        metrics["threat_loss"] = float(eval_results.get("threat_loss", 0.0))
        metrics["escalation_loss"] = float(eval_results.get("escalation_loss", 0.0))

        elapsed = time.time() - start
        self._record_evaluation(metrics, elapsed)

        return (
            metrics["total_loss"],
            len(self.x_test),
            metrics,
        )

    # ───────────────────────────────────────────────────────────────────
    #   Logging
    # ───────────────────────────────────────────────────────────────────

    def _record_training(self, history, elapsed: float) -> None:
        with open(self.training_log, "a") as f:
            f.write(f"Node|{self.node}| Round: {self.round_count}\n")
            f.write(f"Training Time Elapsed: {elapsed} seconds\n")
            for epoch in range(self.epochs):
                f.write(f"Epoch {epoch + 1}/{self.epochs}\n")
                for metric, values in history.history.items():
                    if epoch < len(values):
                        f.write(f"{metric}: {values[epoch]}\n")
                f.write("\n")

    def _record_evaluation(self, metrics: dict, elapsed: float) -> None:
        with open(self.evaluation_log, "a") as f:
            f.write(f"Node|{self.node}| Round: {self.evaluate_count}\n")
            f.write(f"Evaluation Time Elapsed: {elapsed} seconds\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    def save(self, save_name: str, archive_path: str | None = None) -> str:
        """Persist to ``<archive_path>/fed_fusion_mlp_<save_name>.keras``."""
        if archive_path is None:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(
                current_file_dir, "..", "..", "..", "..", "..",
            ))
            archive_path = os.path.join(project_root, "ModelArchive")
        os.makedirs(archive_path, exist_ok=True)
        out_path = os.path.join(archive_path, f"fed_fusion_mlp_{save_name}.keras")
        self.model.save(out_path)
        return out_path
