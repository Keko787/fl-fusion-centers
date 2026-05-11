"""Centralized multi-task MLP trainer for the fusion-centers update.

Phase B.4 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

Trainer contract matches the existing ``CentralNidsClient``: ``__init__``
compiles, ``.fit()`` runs one round of training, ``.evaluate()`` writes
metrics to disk, ``.save(save_name)`` persists under ``ModelArchive/``.

The Y inputs (``y_train`` / ``y_val`` / ``y_test``) are
``(y_class, y_escalation)`` 2-tuples as produced by
``preprocess_communities_crime``. They get re-keyed onto the model's
named output dict ``{"threat": ..., "escalation": ...}`` before being
handed to Keras.
"""
from __future__ import annotations

import os
import time

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import (
    f1_score, mean_absolute_error, precision_recall_fscore_support, roc_auc_score,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


class CentralFusionMLPClient:
    """Centralized trainer for the fusion-centers multi-task MLP."""

    def __init__(self,
                 model,
                 x_train, x_val, x_test,
                 y_train, y_val, y_test,
                 BATCH_SIZE, epochs, steps_per_epoch,
                 learning_rate,
                 num_classes: int = 3,
                 escalation_loss_weight: float = 0.5,
                 evaluation_log: str = "fusion_mlp_eval.log",
                 training_log: str = "fusion_mlp_train.log",
                 node: int = 1,
                 # Callback flags / params (reuse NIDS semantics)
                 earlyStopEnabled=None, lrSchedRedEnabled=None,
                 modelCheckpointEnabled=None,
                 metric_to_monitor_es="val_loss", es_patience=5,
                 restor_best_w=True,
                 metric_to_monitor_l2lr="val_loss", l2lr_patience=3,
                 l2lr_factor=0.1,
                 metric_to_monitor_mc="val_loss",
                 save_best_only=True, checkpoint_mode="min",
                 model_name: str = "fusion_mlp",
                 print_summary: bool = True,
                 shuffle_seed: int = 0):
        # --- store inputs ---
        self.model = model
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        # y_* are (y_class, y_escalation) tuples; unpack once for clarity
        self.y_train_class, self.y_train_esc = y_train
        self.y_val_class, self.y_val_esc = y_val
        self.y_test_class, self.y_test_esc = y_test

        self.batch_size = BATCH_SIZE
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.escalation_loss_weight = float(escalation_loss_weight)

        self.evaluation_log = evaluation_log
        self.training_log = training_log
        self.node = node
        self.model_name = model_name
        self.shuffle_seed = int(shuffle_seed)

        # counters
        self.round_count = 0
        self.evaluate_count = 0

        # --- compile with the joint loss ---
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
        print(f"\n=== CentralFusionMLPClient compiled (β={beta}, "
              f"α={1.0 - beta}, lr={self.learning_rate}) ===")

        # --- assemble callbacks ---
        self.callbacks = []
        if earlyStopEnabled:
            self.callbacks.append(EarlyStopping(
                monitor=metric_to_monitor_es,
                patience=es_patience,
                restore_best_weights=restor_best_w,
            ))
        if lrSchedRedEnabled:
            self.callbacks.append(ReduceLROnPlateau(
                monitor=metric_to_monitor_l2lr,
                patience=l2lr_patience,
                factor=l2lr_factor,
            ))
        if modelCheckpointEnabled:
            self.callbacks.append(ModelCheckpoint(
                filepath=f"best_{self.model_name}.h5",
                monitor=metric_to_monitor_mc,
                mode=checkpoint_mode,
                save_best_only=save_best_only,
            ))

        if print_summary:
            self.model.summary()

    # ───────────────────────────────────────────────────────────────────
    #   Training
    # ───────────────────────────────────────────────────────────────────

    def fit(self):
        self.round_count += 1
        print(f"\n=== CentralFusionMLPClient.fit round={self.round_count} ===")
        start = time.time()

        # Wrap inputs in tf.data.Dataset.repeat() so steps_per_epoch is
        # honored contractually (Keras 3 emits "input ran out of data"
        # warnings when steps_per_epoch is set on raw numpy). Training
        # data uses .shuffle(...).repeat(); validation runs through
        # finite-Dataset once per epoch.
        train_ds = (
            tf.data.Dataset.from_tensor_slices((
                self.x_train,
                {"threat": self.y_train_class,
                 "escalation": self.y_train_esc},
            ))
            .shuffle(buffer_size=len(self.x_train),
                     seed=self.shuffle_seed,
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
            callbacks=self.callbacks,
            verbose=2,
        )

        elapsed = time.time() - start
        self._record_training(history, elapsed)
        return history

    # ───────────────────────────────────────────────────────────────────
    #   Evaluation
    # ───────────────────────────────────────────────────────────────────

    def evaluate(self):
        self.evaluate_count += 1
        print(f"\n=== CentralFusionMLPClient.evaluate round={self.evaluate_count} ===")
        start = time.time()

        threat_pred, escalation_pred = self.model.predict(
            self.x_test, batch_size=self.batch_size, verbose=0,
        )
        elapsed = time.time() - start

        metrics = self._compute_metrics(
            self.y_test_class, threat_pred,
            self.y_test_esc, escalation_pred,
            num_classes=self.num_classes,
        )

        # The total loss the model would compute internally — useful for
        # external comparison against centralized vs federated runs.
        eval_results = self.model.evaluate(
            self.x_test,
            {"threat": self.y_test_class, "escalation": self.y_test_esc},
            batch_size=self.batch_size, verbose=0, return_dict=True,
        )
        metrics["total_loss"] = float(eval_results.get("loss", 0.0))
        metrics["threat_loss"] = float(eval_results.get("threat_loss", 0.0))
        metrics["escalation_loss"] = float(eval_results.get("escalation_loss", 0.0))

        self._record_evaluation(metrics, elapsed)
        return metrics

    @staticmethod
    def _compute_metrics(y_class_true, threat_pred,
                          y_esc_true, escalation_pred,
                          num_classes: int = 3) -> dict:
        """Macro-F1, per-class P/R, escalation MAE, AUROC@median, Spearman.

        Escalation is reported via two complementary metrics:
          * **AUROC at the dataset median** — outline §7.5 frames
            escalation as a ranking task; binarizing at the median
            keeps the test split 50/50 so AUROC is always defined
            (the fixed-0.5 threshold collapses to single-class on
            skewed distributions).
          * **Spearman rank correlation** — the proper ranking metric;
            doesn't require any binarization. Defined whenever both
            sequences have ≥ 2 distinct values.
        Both fall back to ``nan`` for the (rare) degenerate cases.
        """
        y_esc_true = np.asarray(y_esc_true)
        threat_pred_class = np.argmax(threat_pred, axis=1)
        escalation_pred = escalation_pred.flatten()

        labels = list(range(num_classes))
        macro_f1 = f1_score(y_class_true, threat_pred_class,
                             labels=labels, average="macro", zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_class_true, threat_pred_class,
            labels=labels, zero_division=0,
        )
        mae = mean_absolute_error(y_esc_true, escalation_pred)

        # AUROC binarized at the median of the true scores.
        median = float(np.median(y_esc_true))
        y_esc_binary = (y_esc_true >= median).astype(int)
        if len(np.unique(y_esc_binary)) < 2:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(y_esc_binary, escalation_pred)

        # Spearman rank correlation — true ranking metric. Returns nan
        # if either sequence is constant.
        if len(np.unique(y_esc_true)) < 2 or len(np.unique(escalation_pred)) < 2:
            spearman = float("nan")
        else:
            corr, _ = spearmanr(y_esc_true, escalation_pred)
            spearman = float(corr) if not np.isnan(corr) else float("nan")

        # Phase E.3 — overall classification accuracy. Used by the
        # server's fairness-variance calculation; also useful as a
        # standalone per-client metric in the eval log.
        threat_accuracy = float(np.mean(
            threat_pred_class == np.asarray(y_class_true)
        ))

        out: dict = {
            "threat_macro_f1": float(macro_f1),
            "threat_accuracy": threat_accuracy,
        }
        for i in range(num_classes):
            out[f"threat_class_{i}_precision"] = float(precision[i])
            out[f"threat_class_{i}_recall"] = float(recall[i])
        out["escalation_mae"] = float(mae)
        out["escalation_auroc"] = float(auroc)
        out["escalation_spearman"] = spearman
        return out

    # ───────────────────────────────────────────────────────────────────
    #   Save / log
    # ───────────────────────────────────────────────────────────────────

    def save(self, save_name: str, archive_path: str | None = None) -> str:
        """Persist to ``<archive_path>/fusion_mlp_<save_name>.keras``.

        Uses the Keras 3 native ``.keras`` zip format (HDF5 ``.h5`` is
        deprecated upstream). If ``archive_path`` is not given,
        defaults to ``<project_root>/ModelArchive/`` for parity with
        the legacy NIDS / GAN trainer convention. The dispatcher passes
        ``args.run_dir`` so fusion-centers runs keep their model
        artifact next to ``partition_stats.json`` and the evaluation
        log.

        Returns the absolute path written.
        """
        if archive_path is None:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(
                current_file_dir, "..", "..", "..", "..", "..",
            ))
            archive_path = os.path.join(project_root, "ModelArchive")
        os.makedirs(archive_path, exist_ok=True)
        out_path = os.path.join(archive_path, f"fusion_mlp_{save_name}.keras")
        self.model.save(out_path)
        print(f"=== Saved FUSION-MLP to {out_path} ===")
        return out_path

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
