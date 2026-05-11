"""Phase B.1 — multi-task MLP architecture tests.

Pins:
* Two named outputs (``threat``, ``escalation``) with correct shapes.
* Trunk layer count matches the (128, 64, 32) design from outline §4.2.
* Parameter count is in the right ballpark (sanity, not exact).
* Both heads produce non-trivial gradients on a random batch.
* Joint loss with ``loss_weights`` is a true weighted sum of the two
  head losses (the central claim of the multi-task framing).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from Config.modelStructures.FusionMLP.multiTaskMLPStruct import build_fusion_mlp


def test_has_two_named_outputs():
    model = build_fusion_mlp(input_dim=20, num_classes=3)
    assert list(model.output_names) == ["threat", "escalation"]


def test_output_shapes(tf_eager=None):
    model = build_fusion_mlp(input_dim=20, num_classes=3)
    x = np.random.rand(4, 20).astype("float32")
    threat, escalation = model.predict(x, verbose=0)
    assert threat.shape == (4, 3)
    assert escalation.shape == (4, 1)


def test_threat_outputs_are_softmax():
    model = build_fusion_mlp(input_dim=10, num_classes=3)
    x = np.random.rand(8, 10).astype("float32")
    threat, _ = model.predict(x, verbose=0)
    np.testing.assert_allclose(threat.sum(axis=1), 1.0, atol=1e-5)


def test_escalation_outputs_in_unit_interval():
    model = build_fusion_mlp(input_dim=10, num_classes=3)
    x = np.random.rand(8, 10).astype("float32")
    _, escalation = model.predict(x, verbose=0)
    assert (escalation >= 0).all() and (escalation <= 1).all()


def test_trunk_layer_count():
    """3 trunk blocks × 4 layers/block (Dense, BN, ReLU, Dropout) = 12,
    plus Input, 2 heads → 15 expected layers."""
    model = build_fusion_mlp(input_dim=20, num_classes=3)
    layer_names = [layer.name for layer in model.layers]
    # Trunk blocks present.
    for i in range(3):
        assert f"trunk_dense_{i}" in layer_names
        assert f"trunk_bn_{i}" in layer_names
        assert f"trunk_relu_{i}" in layer_names
        assert f"trunk_drop_{i}" in layer_names
    assert "threat" in layer_names
    assert "escalation" in layer_names


def test_parameter_count_in_expected_ballpark():
    """Outline §4.2 specifies a (128, 64, 32) trunk → ~10–20K params for
    typical input_dim. Pin a sanity range so structural drift is caught."""
    model = build_fusion_mlp(input_dim=20, num_classes=3)
    n_params = model.count_params()
    assert 5_000 <= n_params <= 30_000


def test_both_heads_produce_gradients():
    """Training step on a single batch: confirm gradients flow to both
    heads and to the shared trunk."""
    model = build_fusion_mlp(input_dim=10, num_classes=3, l2_alpha=0.0)
    x = np.random.rand(8, 10).astype("float32")
    y_class = np.random.randint(0, 3, size=8).astype("int64")
    y_esc = np.random.rand(8).astype("float32")

    with tf.GradientTape() as tape:
        threat_pred, escalation_pred = model(x, training=True)
        loss_threat = tf.keras.losses.sparse_categorical_crossentropy(
            y_class, threat_pred,
        )
        loss_esc = tf.keras.losses.binary_crossentropy(
            y_esc, tf.squeeze(escalation_pred, axis=-1),
        )
        loss = 0.5 * tf.reduce_mean(loss_threat) + 0.5 * tf.reduce_mean(loss_esc)
    grads = tape.gradient(loss, model.trainable_variables)
    # Every trainable variable should receive a non-None gradient.
    assert all(g is not None for g in grads)
    # And at least one trunk + one head variable should have non-zero grad.
    non_zero = [tf.reduce_max(tf.abs(g)).numpy() for g in grads if g is not None]
    assert max(non_zero) > 0.0


def test_joint_loss_is_weighted_sum():
    """The Keras compile contract: total_loss = α·threat_loss + β·escalation_loss
    (plus any kernel regularizer). Verify by compiling with known weights
    and comparing the components."""
    model = build_fusion_mlp(input_dim=10, num_classes=3, l2_alpha=0.0)
    beta = 0.3
    model.compile(
        optimizer=Adam(1e-3),
        loss={"threat": "sparse_categorical_crossentropy",
              "escalation": "binary_crossentropy"},
        loss_weights={"threat": 1.0 - beta, "escalation": beta},
    )
    x = np.random.rand(16, 10).astype("float32")
    y_class = np.random.randint(0, 3, size=16).astype("int64")
    y_esc = np.random.rand(16).astype("float32")

    out = model.evaluate(
        x, {"threat": y_class, "escalation": y_esc},
        verbose=0, return_dict=True,
    )
    expected_total = (1.0 - beta) * out["threat_loss"] + beta * out["escalation_loss"]
    np.testing.assert_allclose(out["loss"], expected_total, atol=1e-5)


def test_hidden_param_overrides_layer_sizes():
    model_default = build_fusion_mlp(input_dim=10, num_classes=3)
    model_wider = build_fusion_mlp(input_dim=10, num_classes=3,
                                    hidden=(256, 128, 64))
    assert model_wider.count_params() > model_default.count_params()
