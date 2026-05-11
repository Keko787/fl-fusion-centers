"""Phase B.2 — hyperparameterLoading FUSION-MLP branch tests.

Pins:
* The branch returns the same 20-tuple shape as every other model
  type (no positional drift).
* Defaults match the implementation plan B.2 (batch_size=128, lr=1e-3,
  num_classes=3, l2_alpha=1e-4).
* steps_per_epoch floors at 1 so tiny synthetic partitions still train.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from Config.SessionConfig.hyperparameterLoading import hyperparameterLoading


def _args_for_fusion(regularization=True, dp=False, es=False, lr=False, mc=False):
    return SimpleNamespace(
        model_type="FUSION-MLP",
        DP_enabled=dp,
        earlyStopEnabled=es,
        lrSchedRedEnabled=lr,
        modelCheckpointEnabled=mc,
    )


def test_fusion_mlp_branch_returns_expected_defaults():
    args = _args_for_fusion()
    x_train = np.random.rand(1000, 20).astype("float32")
    result = hyperparameterLoading(args, x_train)
    assert len(result) == 20

    (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes,
     latent_dim, betas, learning_rate, l2_alpha, l2_norm_clip,
     noise_multiplier, num_microbatches, *_) = result

    assert BATCH_SIZE == 128
    assert noise_dim is None  # No GAN component
    assert input_dim == 20
    assert num_classes == 3
    assert latent_dim is None
    assert learning_rate == 1e-3
    assert betas == [0.9, 0.999]
    assert steps_per_epoch == 1000 // 128


def test_fusion_mlp_l2_alpha_when_regularization_enabled():
    args = _args_for_fusion(regularization=True)
    x_train = np.random.rand(500, 15).astype("float32")
    result = hyperparameterLoading(args, x_train)
    l2_alpha = result[8]  # 9th slot in the canonical tuple
    # The shared regularizationEnabled flag is hardcoded True at the top
    # of hyperparameterLoading, so this branch always returns l2_alpha=1e-4.
    assert l2_alpha == 1e-4


def test_fusion_mlp_steps_per_epoch_floors_at_one():
    """Tiny partition (< batch_size rows) must still produce ≥ 1 step."""
    args = _args_for_fusion()
    x_train = np.random.rand(50, 10).astype("float32")
    result = hyperparameterLoading(args, x_train)
    BATCH_SIZE, _, steps_per_epoch, *_ = result
    assert steps_per_epoch >= 1


def test_fusion_mlp_callback_params_when_enabled():
    args = _args_for_fusion(es=True, lr=True, mc=True)
    x_train = np.random.rand(500, 15).astype("float32")
    result = hyperparameterLoading(args, x_train)
    (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes,
     latent_dim, betas, learning_rate, l2_alpha, l2_norm_clip,
     noise_multiplier, num_microbatches,
     metric_to_monitor_es, es_patience, restor_best_w,
     metric_to_monitor_l2lr, l2lr_patience, save_best_only,
     metric_to_monitor_mc, checkpoint_mode) = result

    assert metric_to_monitor_es == "val_loss"
    assert es_patience == 5
    assert restor_best_w is True
    assert metric_to_monitor_l2lr == "val_loss"
    assert l2lr_patience == 3
    assert metric_to_monitor_mc == "val_loss"
    assert save_best_only is True
    assert checkpoint_mode == "min"
