"""Multi-task MLP for the fusion-centers FL update (Phase B.1 + Phase D.2).

Shared trunk + two task-specific heads, matching design doc §3.4 and
implementation plan B.1:

    Input(input_dim)
       ↓
    Dense(128, ReLU) → BN → Dropout
    Dense( 64, ReLU) → BN → Dropout
    Dense( 32, ReLU) → BN → Dropout
       ↓
   ┌───────────────┴───────────────┐
   ↓                               ↓
 Dense(num_classes, softmax)     Dense(1, sigmoid)
     name="threat"                  name="escalation"

Two builders:
  * :func:`build_fusion_mlp` — plain ``tf.keras.Model``. Used by the
    centralized trainer (Phase B) and the FedAvg client (Phase C).
  * :func:`build_fedprox_fusion_mlp` — :class:`FedProxFusionMLPModel`
    subclass with FedProx proximal-term injection in ``train_step``.
    Used by the FedProx client (Phase D).

Both share :func:`_build_layers` so architectural changes propagate.
The FedProx subclass behaves identically to the base when
``fedprox_mu == 0`` or before ``set_global_weights`` is called — the
proximal term is gated by both conditions.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization, Dense, Dropout, Input, ReLU,
)
from tensorflow.keras.regularizers import l2


def _build_layers(input_dim: int, num_classes: int,
                  hidden: Sequence[int], dropout: float, l2_alpha: float):
    """Internal: build the shared layer graph; return ``(inputs, [threat, escalation])``."""
    inputs = Input(shape=(input_dim,), name="features")
    reg = l2(l2_alpha) if l2_alpha and l2_alpha > 0 else None

    x = inputs
    for i, units in enumerate(hidden):
        x = Dense(units, kernel_regularizer=reg,
                  name=f"trunk_dense_{i}")(x)
        x = BatchNormalization(name=f"trunk_bn_{i}")(x)
        x = ReLU(name=f"trunk_relu_{i}")(x)
        x = Dropout(dropout, name=f"trunk_drop_{i}")(x)

    threat = Dense(num_classes, activation="softmax",
                   kernel_regularizer=reg, name="threat")(x)
    escalation = Dense(1, activation="sigmoid",
                       kernel_regularizer=reg, name="escalation")(x)
    return inputs, [threat, escalation]


def build_fusion_mlp(input_dim: int,
                     num_classes: int = 3,
                     hidden: Sequence[int] = (128, 64, 32),
                     dropout: float = 0.2,
                     l2_alpha: float = 1e-4) -> Model:
    """Build the multi-task MLP with the base ``tf.keras.Model`` class."""
    inputs, outputs = _build_layers(input_dim, num_classes, hidden, dropout, l2_alpha)
    return Model(inputs=inputs, outputs=outputs, name="fusion_mlp")


def load_fusion_mlp(path: str) -> Model:
    """Re-load a saved fusion MLP. Thin wrapper around ``tf.keras.models.load_model``.

    FedProx-trained models reload as a plain ``tf.keras.Model`` (the
    proximal subclass is training-only state). Inference and evaluation
    work identically; if you need to *resume training* in FedProx mode
    you'll rebuild a fresh ``FedProxFusionMLPModel`` and call
    ``set_weights(...)``.
    """
    return tf.keras.models.load_model(path)


class FedProxFusionMLPModel(tf.keras.Model):
    """tf.keras.Model that adds a FedProx proximal penalty in ``train_step``.

    Phase D.2 of the fusion-centers implementation plan.

    The proximal term is

        (mu / 2) * Σ_i ||w_i - g_i||²

    summed over the model's **trainable** weights, where ``g_i`` is the
    server-broadcast "global" weight set most recently via
    :meth:`set_global_weights`. Until that call (or while ``fedprox_mu``
    is zero), the model behaves identically to the base Functional
    model — the train_step matches Keras's default plus a no-op branch.

    Public state the FL trainer flips per round:
      * ``fedprox_mu`` — current proximal coefficient. Read/write via
        the Python attr; backed by a ``tf.Variable`` so updates are
        picked up across rounds without retracing ``train_step``
        (Phase D review #2).
      * ``set_global_weights(weights)`` — refresh the anchor. Pass the
        **full** weights list (matching ``model.get_weights()`` length,
        i.e. the same list Flower's ``parameters_to_ndarrays`` returns);
        the model internally filters to the trainable subset so
        anchors align with ``self.trainable_weights`` element-wise.

    Reported-loss semantics (Phase D review #3): the ``loss`` value in
    ``model.fit`` history reflects the base + L2 regularization, NOT
    the proximal contribution. The proximal term affects gradients
    (which is what FedProx needs) but is not added to the loss metric
    Keras's ``compute_metrics`` reads from the internal tracker. The
    server-side aggregation uses eval-time ``total_loss`` from
    ``model.evaluate`` which is also base + L2 — so cross-strategy
    comparison stays apples-to-apples.

    To let users *see* what the proximal term is doing, the model
    exposes a sibling ``proximal_contribution`` metric per training
    step (Phase D follow-up). It is the post-scale value
    ``(mu/2) * Σ||w - g||²`` and lands in ``history.history``
    automatically. Zero when ``mu == 0`` or no anchor is set.
    """

    def __init__(self, *args, mu: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # tf.Variable so post-trace updates work (review #2). Wrapped
        # in tf.Variable rather than left as a Python float because
        # Keras 3 tf.functions captures Python values at trace time.
        self._fedprox_mu_var = tf.Variable(
            float(mu), trainable=False, dtype=tf.float32, name="fedprox_mu",
        )
        # Anchor state — populated by set_global_weights(). Stored on
        # ``__dict__`` directly to bypass Keras's variable tracker so
        # the anchors don't get serialized into the .keras artifact
        # alongside the model's own weights (review #6).
        self.__dict__["_global_weight_vars"] = None
        self.__dict__["_trainable_indices"] = None

    @property
    def fedprox_mu(self) -> float:
        """Current proximal coefficient (Python float)."""
        return float(self._fedprox_mu_var.numpy())

    @fedprox_mu.setter
    def fedprox_mu(self, value: float) -> None:
        self._fedprox_mu_var.assign(float(value))

    @property
    def _global_weights_set(self) -> bool:
        """True iff :meth:`set_global_weights` has been called."""
        return self.__dict__.get("_global_weight_vars") is not None

    def set_global_weights(self, weights: Iterable) -> None:
        """Cache the server-broadcast weights as the proximal anchor.

        ``weights`` must match ``len(model.get_weights())`` — the full
        list including non-trainable BN moving statistics. The model
        filters to ``trainable_weights`` indices internally so the
        anchor list aligns element-wise with ``self.trainable_weights``
        (review #1). Passing a wrong-length list raises ``ValueError``
        (review #9).

        Allocates one non-trainable ``tf.Variable`` per trainable layer
        weight on the first call; ``.assign`` on subsequent calls so
        the memory footprint stays flat across rounds.
        """
        weights = list(weights)
        expected = len(self.weights)
        if len(weights) != expected:
            raise ValueError(
                f"set_global_weights expected {expected} weights matching "
                f"model.get_weights() (the full list including non-trainable "
                f"BN moving statistics), but got {len(weights)}. The "
                f"proximal term operates on the trainable subset; pass "
                f"the full list and the model filters internally."
            )

        # Compute the trainable-index mask once and cache it.
        if self.__dict__.get("_trainable_indices") is None:
            trainable_ids = {id(w) for w in self.trainable_weights}
            self.__dict__["_trainable_indices"] = [
                i for i, w in enumerate(self.weights) if id(w) in trainable_ids
            ]
        trainable_indices = self.__dict__["_trainable_indices"]
        anchor_values = [weights[i] for i in trainable_indices]

        if self.__dict__.get("_global_weight_vars") is None:
            anchors = [
                tf.Variable(w, trainable=False, dtype=tf.float32,
                             name=f"global_anchor_{i}")
                for i, w in enumerate(anchor_values)
            ]
            self.__dict__["_global_weight_vars"] = anchors
        else:
            for var, value in zip(self.__dict__["_global_weight_vars"],
                                    anchor_values):
                var.assign(value)

    def train_step(self, data):
        x, y = data
        anchors = self.__dict__.get("_global_weight_vars")
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
            if anchors is not None:
                proximal = tf.add_n([
                    tf.reduce_sum(tf.square(w - g))
                    for w, g in zip(self.trainable_weights, anchors)
                ])
                # tf.where so mid-experiment mu changes are honored
                # without retracing — Phase D review #2.
                proximal_contribution = tf.where(
                    self._fedprox_mu_var > 0,
                    (self._fedprox_mu_var / 2.0) * proximal,
                    tf.zeros_like(proximal),
                )
                loss = loss + proximal_contribution
            else:
                # No anchor set yet (FedAvg path or pre-first-broadcast
                # FedProx) → the proximal contribution is structurally
                # zero. We still emit the metric so the history dict
                # has a consistent schema across strategies.
                proximal_contribution = tf.constant(0.0, dtype=tf.float32)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        metrics = self.compute_metrics(x, y, y_pred, sample_weight=None)
        metrics["proximal_contribution"] = proximal_contribution
        return metrics


def build_fedprox_fusion_mlp(input_dim: int,
                              num_classes: int = 3,
                              hidden: Sequence[int] = (128, 64, 32),
                              dropout: float = 0.2,
                              l2_alpha: float = 1e-4,
                              mu: float = 0.0) -> FedProxFusionMLPModel:
    """Build the multi-task MLP with the FedProx subclass.

    The proximal term is gated by both ``mu > 0`` and a prior
    :meth:`set_global_weights` call, so passing ``mu=0`` (the FedAvg
    default for the broadcast) produces a model that behaves
    identically to :func:`build_fusion_mlp`.
    """
    inputs, outputs = _build_layers(input_dim, num_classes, hidden, dropout, l2_alpha)
    return FedProxFusionMLPModel(
        inputs=inputs, outputs=outputs, name="fedprox_fusion_mlp", mu=mu,
    )
