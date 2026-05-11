#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
from datetime import datetime
import numpy as np
from numpy import expand_dims
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../../..'))

# TensorFlow & Flower
if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# other plugins
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Legacy NIDS trainer depends on tensorflow_privacy, which is incompatible
# with TF 2.21 used by the Fusion Centers stack. Defer the import so the
# FUSION-MLP code path stays runnable on a fusion-only install. The
# symbol becomes a sentinel that raises a clear error iff someone
# actually selects --model_type NIDS.
try:
    from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.NIDS.nidsModelCentralTrainingConfig import CentralNidsClient, recordConfig
    _LEGACY_NIDS_CENTRAL_AVAILABLE = True
    _LEGACY_NIDS_CENTRAL_ERROR = None
except ImportError as _nids_central_import_error:
    _LEGACY_NIDS_CENTRAL_AVAILABLE = False
    _LEGACY_NIDS_CENTRAL_ERROR = str(_nids_central_import_error)

    def _legacy_nids_central_unavailable(*_args, **_kwargs):
        raise ImportError(
            "Legacy NIDS centralized trainer requires tensorflow_privacy. "
            "Install AppSetup/requirements_core.txt for the NIDS code path "
            "(note: this pins TF to 2.15 and is incompatible with the Fusion "
            f"Centers stack). Original import error: {_LEGACY_NIDS_CENTRAL_ERROR}"
        )

    CentralNidsClient = _legacy_nids_central_unavailable
    recordConfig = _legacy_nids_central_unavailable

from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Generator.generatorModelCentralTrainingConfig import CentralGenerator
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.discriminatorBinaryCentralTrainingConfig import CentralBinaryDiscriminator
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.GANBinaryCentralTrainingConfig import CentralBinaryGan
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.WGANBinaryCentralTrainingConfig import CentralBinaryWGan
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Generator.WGenBinaryCentralTrainingConfig import CentralBinaryWGen
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.WDiscBinaryCentralTrainingConfig import CentralBinaryWDisc
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.ACGANCentralTrainingConfig import CentralACGan
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Generator.ACGenCentralTrainingConfig import CentralACGenerator
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.ACDiscREALCentralTrainingConfig import CentralACDiscREAL
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.Discriminator.ACDiscCentralTrainingConfig import CentralACDisc
from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.FusionMLP.fusionMLPCentralTrainingConfig import CentralFusionMLPClient

# from Config.ModelTrainingConfig.ClientModelTrainingConfig.CentralTrainingConfig.GAN.FullModel.CANGANCentralTrainingConfig import CANGAN

################################################################################################################
#                      CENTRAL/Local TRAINING CONFIG CLIENT CLASS OBJECT LOADER                                #
################################################################################################################

def modelCentralTrainingConfigLoad(nids, discriminator, generator, GAN, dataset_used, model_type, train_type,
                                   earlyStopEnabled, DP_enabled, lrSchedRedEnabled, modelCheckpointEnabled, X_train_data,
                                   X_val_data, y_train_data, y_val_data, X_test_data, y_test_data, node, BATCH_SIZE,
                                   epochs, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas,
                                   learning_rate, l2_alpha, l2_norm_clip, noise_multiplier, num_microbatches,
                                   metric_to_monitor_es, es_patience, restor_best_w, metric_to_monitor_l2lr,
                                   l2lr_patience, save_best_only, metric_to_monitor_mc, checkpoint_mode, evaluationLog,
                                   trainingLog, args=None):

    client = None

    if model_type == 'NIDS':
        client = CentralNidsClient(nids, dataset_used, node, earlyStopEnabled, DP_enabled,
                                   lrSchedRedEnabled, modelCheckpointEnabled, X_train_data, y_train_data, X_test_data,
                                   y_test_data, X_val_data, y_val_data, l2_norm_clip, noise_multiplier,
                                   num_microbatches,
                                   BATCH_SIZE, epochs, steps_per_epoch, learning_rate, metric_to_monitor_es,
                                   es_patience, restor_best_w, metric_to_monitor_l2lr, l2lr_patience, save_best_only,
                                   metric_to_monitor_mc, checkpoint_mode, evaluationLog, trainingLog)

    # if models are adv nids
    elif model_type == "NIDS-IOT-Binary":
        client = None

    elif model_type == "NIDS-IOT-Multiclass":
        client = None

    elif model_type == "NIDS-IOT-Multiclass-Dynamic":
        client = None

    # if model are the GAN types
    elif model_type == 'GAN':
        if train_type == "Both":
            client = CentralBinaryGan(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                  y_test_data, BATCH_SIZE,
                                  noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = CentralGenerator(generator, discriminator, X_train_data, X_val_data, y_train_data, y_val_data,
                                      X_test_data, y_test_data, BATCH_SIZE,
                                      noise_dim, epochs, steps_per_epoch)

        elif train_type == "Discriminator":
            client = CentralBinaryDiscriminator(discriminator, generator, X_train_data, X_val_data, y_train_data, y_val_data,
                                          X_test_data, y_test_data, BATCH_SIZE, noise_dim, epochs, steps_per_epoch)


    elif model_type == 'WGAN-GP':
        if train_type == "Both":
            client = CentralBinaryWGan(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = CentralBinaryWGen(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Discriminator":
            client = CentralBinaryWDisc(GAN, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data,
                                   y_test_data, BATCH_SIZE,
                                   noise_dim, epochs, steps_per_epoch, learning_rate)

    elif model_type == 'FUSION-MLP':
        # Phase B.5 — centralized multi-task MLP trainer. ``nids`` holds
        # the FUSION-MLP per the modelCreateLoad slot convention.
        # Fusion-specific kwargs are pulled from ``args`` to keep the
        # legacy positional tuple stable for every other trainer.
        if args is None:
            raise ValueError(
                "FUSION-MLP centralized trainer requires args (for "
                "escalation_loss_weight). Pass args=args from the entry point."
            )
        client = CentralFusionMLPClient(
            model=nids,
            x_train=X_train_data, x_val=X_val_data, x_test=X_test_data,
            y_train=y_train_data, y_val=y_val_data, y_test=y_test_data,
            BATCH_SIZE=BATCH_SIZE, epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            learning_rate=learning_rate,
            num_classes=num_classes,
            escalation_loss_weight=args.escalation_loss_weight,
            shuffle_seed=getattr(args, "commcrime_random_seed", 0),
            evaluation_log=evaluationLog,
            training_log=trainingLog,
            node=node,
            earlyStopEnabled=earlyStopEnabled,
            lrSchedRedEnabled=lrSchedRedEnabled,
            modelCheckpointEnabled=modelCheckpointEnabled,
            metric_to_monitor_es=metric_to_monitor_es,
            es_patience=es_patience or 5,
            restor_best_w=restor_best_w if restor_best_w is not None else True,
            metric_to_monitor_l2lr=metric_to_monitor_l2lr,
            l2lr_patience=l2lr_patience or 3,
            metric_to_monitor_mc=metric_to_monitor_mc,
            save_best_only=save_best_only if save_best_only is not None else True,
            checkpoint_mode=checkpoint_mode or "min",
        )

    elif model_type == 'AC-GAN':
        if train_type == "Both":
            client = CentralACGan(discriminator, generator, nids, X_train_data, X_val_data, y_train_data,
                              y_val_data, X_test_data, y_test_data, BATCH_SIZE,
                              noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate)
        elif train_type == "Generator":
            client = CentralACGenerator(discriminator, generator, nids, X_train_data, X_val_data, y_train_data,
                                        y_val_data, X_test_data, y_test_data, BATCH_SIZE, noise_dim, latent_dim,
                                        num_classes, input_dim, epochs, steps_per_epoch)
        elif train_type == "Discriminator":
            client = CentralACDiscREAL(discriminator=discriminator,
                                       generator=generator,
                                       nids=nids,
                                       x_train=X_train_data,
                                       x_val=X_val_data,
                                       y_train=y_train_data,
                                       y_val=y_val_data,
                                       x_test=X_test_data,
                                       y_test=y_test_data,
                                       BATCH_SIZE=BATCH_SIZE,
                                       noise_dim=noise_dim,
                                       latent_dim=latent_dim,
                                       num_classes=num_classes,
                                       input_dim=input_dim,
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       learning_rate=learning_rate,
                                       use_class_labels=True,
                                       log_file=trainingLog)
            # optionally use discriminator that uses centralized training with fake data as well
            # client = CentralACDisc(discriminator=discriminator,
            #                       generator=generator,
            #                       nids=nids,
            #                       x_train=X_train_data,
            #                       x_val=X_val_data,
            #                       y_train=y_train_data,
            #                       y_val=y_val_data,
            #                       x_test=X_test_data,
            #                       y_test=y_test_data,
            #                       BATCH_SIZE=BATCH_SIZE,
            #                       noise_dim=noise_dim,
            #                       latent_dim=latent_dim,
            #                       num_classes=num_classes,
            #                       input_dim=input_dim,
            #                       epochs=epochs,
            #                       steps_per_epoch=steps_per_epoch,
            #                       learning_rate=learning_rate,
            #                       log_file=trainingLog)

        # elif model_type == 'CANGAN':
        #     if train_type == "Both":
        #         # client = CentralCANGAN(discriminator, generator, nids, X_train_data, X_val_data, y_train_data,
        #         #                       y_val_data, X_test_data, y_test_data, BATCH_SIZE,
        #         #                       noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch,
        #         #                       learning_rate)
            # elif train_type == "Generator":
            #
            # elif train_type == "Discriminator":


    return client
