"""Server-side FedAvg strategy for the fusion-centers FL update.

Phase C.3 of `DeveloperDocs/Fusion_Centers_FL_Update_Implementation_Plan.md`.

Extends :class:`fl.server.strategy.FedAvg` with:
  * **Weighted aggregation of paper-figure metrics** — macro-F1,
    accuracy, escalation MAE, AUROC, Spearman — weighted by per-client
    ``num_examples``. The base FedAvg only aggregates ``loss``.
  * **Per-client accuracy variance** as the **canonical fairness
    metric** (outline §7.5: "variance of per-client accuracy on the
    global test set"). The macro-F1 variant
    ``fairness_macro_f1_variance`` is also emitted as a secondary
    measure (class-imbalance-sensitive) but plot scripts and paper
    figures should default to ``fairness_accuracy_variance`` per the
    outline.
  * **Per-round summary log** written to disk so Phase E plot scripts
    can reconstruct convergence curves from one file.
  * **Plateau detection** — flags when aggregated ``val_loss`` has not
    improved by ≥ ``plateau_tol`` for ``plateau_patience`` consecutive
    rounds. The simulation completes the full round budget either way;
    early-termination of simulation is deferred to Phase D.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class FusionFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with fusion-centers metric aggregation + plateau detection."""

    def __init__(self, *,
                 evaluation_log: Optional[str] = None,
                 plateau_patience: int = 10,
                 plateau_tol: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.evaluation_log = evaluation_log
        self.plateau_patience = int(plateau_patience)
        self.plateau_tol = float(plateau_tol)

        self._best_val_loss: float = float("inf")
        self._rounds_since_improvement: int = 0
        self.plateau_detected: bool = False
        # Latest aggregated parameters captured during ``aggregate_fit``.
        # Used by :func:`SimulationRunner._save_final_model` to write the
        # actual trained model artifact (not the initial weights — the
        # bug Phase C review #1 caught).
        self._final_parameters: Optional[Parameters] = None
        # Per-round history kept on the instance so the integration test
        # can inspect aggregation behavior without parsing the log file.
        self.history: List[Dict[str, float]] = []
        # Phase E.2 — federation-overhead instrumentation. ``configure_fit``
        # stamps the round's start time in ``_round_start_times[round]``;
        # ``aggregate_fit`` computes ``round_seconds`` against it and
        # the per-client byte sum of the uploaded weights. Stashed on
        # the instance so ``aggregate_evaluate`` can log them alongside
        # the round's eval metrics.
        #
        # Phase E review #8 — keyed by ``server_round`` rather than a
        # single scalar so that Flower's retry-on-transient-failure
        # (which re-invokes ``configure_fit`` for the same round) does
        # NOT overwrite the original start time. ``setdefault`` keeps
        # the first stamp for a given round.
        self._round_start_times: Dict[int, float] = {}
        self._fit_overhead: Dict[str, float] = {}

    # ───────────────────────────────────────────────────────────────────
    #   Fit-side instrumentation (Phase E.2 federation-overhead metrics)
    # ───────────────────────────────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Stamp the round's start time so ``aggregate_fit`` can compute wall-clock.

        Uses ``setdefault`` so Flower's transient-failure retry (which
        re-invokes ``configure_fit`` for the same ``server_round``)
        does NOT overwrite the original stamp — ``round_seconds`` thus
        captures total round wall-clock, not just the post-retry slice.
        """
        self._round_start_times.setdefault(server_round, time.time())
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Stash final parameters + compute federation-overhead metrics.

        * ``_final_parameters``: latest aggregated weights (Phase C review #1).
        * ``round_seconds``: wall-clock from ``configure_fit`` start to
          this aggregation finish — covers training + return-trip.
        * ``parameter_update_wire_bytes``: sum of per-client on-wire byte
          counts. Flower's ``Parameters.tensors`` are pre-serialized
          ``bytes`` objects so the count is the exact post-serialization
          payload size. This is the federation-overhead measurement
          (captures any future compression / encoding); NOT the raw
          ``4 × n_params × n_clients`` float32 byte count.
        * ``proximal_contribution``: weighted mean of clients' fit-side
          proximal contribution (FedProx emits a positive value;
          plain-FedAvg clients emit 0). Phase E review #4.

        Both overhead metrics are stashed on the instance and logged at
        the next ``aggregate_evaluate`` call (single log file, single
        row per round).
        """
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self._final_parameters = params

        # Phase E.2 — overhead measurements.
        #
        # ``parameter_update_wire_bytes`` counts the on-wire byte size
        # of each client's serialized weights (Flower's
        # ``Parameters.tensors`` is already post-serialization bytes).
        # This is the federation-overhead measurement we want for paper
        # figures — it captures serialization overhead and any future
        # compression. NOT the raw float32 weight byte count, which
        # would be ``4 × n_params × n_clients``.
        start_time = self._round_start_times.get(server_round)
        round_seconds = (
            float(time.time() - start_time)
            if start_time is not None else 0.0
        )
        total_wire_bytes = 0
        for _, fit_res in results:
            for tensor_bytes in fit_res.parameters.tensors:
                total_wire_bytes += len(tensor_bytes)

        # Phase E review #4 — aggregate ``proximal_contribution`` across
        # clients so the FL server log carries the FedProx penalty's
        # evolution. Plain-FedAvg clients emit 0.0 (consistent schema
        # per Phase D follow-up); FedProx clients emit the actual
        # ``(μ/2)·Σ‖w-g‖²`` value.
        prox_total_examples = sum(r.num_examples for _, r in results)
        prox_weighted = (
            sum(r.num_examples * float(r.metrics.get("proximal_contribution", 0.0))
                for _, r in results) / prox_total_examples
            if prox_total_examples > 0 else 0.0
        )

        self._fit_overhead = {
            "round_seconds": round_seconds,
            "parameter_update_wire_bytes": float(total_wire_bytes),
            "parameter_update_wire_bytes_per_client": (
                float(total_wire_bytes) / len(results) if results else 0.0
            ),
            "proximal_contribution": float(prox_weighted),
        }
        # Also return overhead in the fit metrics dict so Flower's History
        # captures it for callers that introspect ``History.metrics_distributed_fit``.
        out_metrics = dict(metrics) if metrics else {}
        out_metrics.update(self._fit_overhead)
        return params, out_metrics

    # ───────────────────────────────────────────────────────────────────
    #   Evaluate-side aggregation (fusion-centers metric weighted means)
    # ───────────────────────────────────────────────────────────────────

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Standard FedAvg aggregation gives a weighted-mean loss.
        loss_aggregated, _ = super().aggregate_evaluate(
            server_round, results, failures,
        )

        if not results:
            return loss_aggregated, {}

        # Weighted means for the metrics our trainer reports.
        # ``threat_accuracy`` added Phase E review #3 so plot scripts can
        # show the aggregated accuracy alongside its fairness variance.
        weighted_metrics_keys = (
            "threat_macro_f1",
            "threat_accuracy",
            "escalation_mae",
            "escalation_auroc",
            "escalation_spearman",
            "total_loss",
            "threat_loss",
            "escalation_loss",
        )
        aggregated: Dict[str, float] = {}
        total_examples = sum(r.num_examples for _, r in results)
        if total_examples <= 0:
            return loss_aggregated, {}

        for key in weighted_metrics_keys:
            # Filter clients reporting NaN (e.g. single-class AUROC
            # fallback) so they don't bias the weighted mean toward 0.
            # Use the raw float — _safe_float would coerce NaN to 0
            # which is exactly the silent-bias bug we want to avoid.
            valid = [(r.num_examples, float(r.metrics.get(key, float("nan"))))
                     for _, r in results
                     if r.metrics.get(key) is not None
                     and not _is_nan(r.metrics.get(key))]
            if not valid:
                aggregated[key] = float("nan")
            else:
                valid_total = sum(w for w, _ in valid)
                aggregated[key] = _weighted_mean(valid, valid_total)

        # ── Fairness metrics ────────────────────────────────────────
        # Canonical: ``fairness_accuracy_variance`` (Phase E review #7).
        # Outline §7.5 literally says "variance of per-client accuracy on
        # the global test set" — that's the paper-figure fairness number.
        # ``fairness_macro_f1_variance`` is emitted as a secondary, more
        # class-imbalance-sensitive measure; the two converge on
        # well-balanced data and diverge on skewed partitions.
        #
        # Phase E review #1 — filter NaN per-client values rather than
        # coercing them to 0 (the bug ``_safe_float`` introduced). A
        # client reporting NaN for any metric → drop from the variance
        # calculation; if fewer than 2 valid values remain, variance is
        # undefined and we return 0 (consistent with prior behavior).
        per_client_acc = [
            float(r.metrics.get("threat_accuracy"))
            for _, r in results
            if r.metrics.get("threat_accuracy") is not None
            and not _is_nan(r.metrics.get("threat_accuracy"))
        ]
        aggregated["fairness_accuracy_variance"] = (  # canonical
            float(_variance(per_client_acc)) if len(per_client_acc) > 1 else 0.0
        )
        per_client_f1 = [
            float(r.metrics.get("threat_macro_f1"))
            for _, r in results
            if r.metrics.get("threat_macro_f1") is not None
            and not _is_nan(r.metrics.get("threat_macro_f1"))
        ]
        aggregated["fairness_macro_f1_variance"] = (  # secondary
            float(_variance(per_client_f1)) if len(per_client_f1) > 1 else 0.0
        )

        # Plateau detection on aggregated loss.
        agg_loss = float(loss_aggregated) if loss_aggregated is not None else float("nan")
        if agg_loss + self.plateau_tol < self._best_val_loss:
            self._best_val_loss = agg_loss
            self._rounds_since_improvement = 0
        else:
            self._rounds_since_improvement += 1
        if (not self.plateau_detected
                and self._rounds_since_improvement >= self.plateau_patience):
            self.plateau_detected = True
            print(f"⚠️  FusionFedAvg: plateau detected at round {server_round} "
                  f"({self._rounds_since_improvement} rounds since improvement, "
                  f"best={self._best_val_loss:.6f})")

        aggregated["plateau_detected"] = float(self.plateau_detected)
        aggregated["rounds_since_improvement"] = float(self._rounds_since_improvement)

        # Phase E.2 — fold in fit-side overhead metrics captured during
        # this round's aggregate_fit. The aggregate_evaluate path is the
        # canonical "per-round summary line" the plot scripts read, so
        # we duplicate the overhead here for downstream convenience.
        aggregated.update(self._fit_overhead)

        # Stash + log.
        self.history.append({"round": server_round, "loss": agg_loss, **aggregated})
        if self.evaluation_log:
            self._append_log(server_round, agg_loss, aggregated, len(results))

        return loss_aggregated, aggregated

    # ───────────────────────────────────────────────────────────────────
    #   Logging
    # ───────────────────────────────────────────────────────────────────

    def _append_log(self, server_round: int, agg_loss: float,
                     metrics: Dict[str, float], n_clients: int) -> None:
        Path(self.evaluation_log).parent.mkdir(parents=True, exist_ok=True)
        with open(self.evaluation_log, "a") as f:
            f.write(f"=== Round {server_round} ({n_clients} clients) ===\n")
            f.write(f"aggregated_loss: {agg_loss}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")


def _safe_float(value) -> float:
    """Coerce to float, treating NaN/None as 0. Used only for the
    fairness variance metric where NaN inputs should not crash the
    calculation. Metric aggregation paths that care about NaN
    (paper-figure metrics) filter via :func:`_is_nan` before this."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def _is_nan(value) -> bool:
    """True if ``value`` is None or NaN (Python float or numpy)."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _weighted_mean(weighted_values, total_weight: float) -> float:
    if total_weight <= 0:
        return float("nan")
    return sum(w * v for w, v in weighted_values) / total_weight


def _variance(values) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
