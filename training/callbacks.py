"""
=========================================================
Training Callbacks
CNN + BiLSTM + CTC OCR
=========================================================
"""

import os

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)

from ocr.config import *


##############################################################
# Create Required Directories
##############################################################

CHECKPOINT_DIR = "training/checkpoints"

LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)


##############################################################
# Model Checkpoint
##############################################################

def checkpoint_callback():

    return ModelCheckpoint(

        filepath=os.path.join(
            CHECKPOINT_DIR,
            "best_model.keras"
        ),

        monitor="val_loss",

        save_best_only=True,

        save_weights_only=False,

        verbose=1
    )


##############################################################
# Early Stopping
##############################################################

def early_stopping_callback():

    return EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True,

        verbose=1
    )


##############################################################
# Learning Rate Scheduler
##############################################################

def learning_rate_callback():

    return ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-6,

        verbose=1
    )


##############################################################
# CSV Logger
##############################################################

def csv_logger_callback():

    return CSVLogger(

        os.path.join(

            LOG_DIR,

            "training_log.csv"

        ),

        append=True
    )


##############################################################
# TensorBoard
##############################################################

def tensorboard_callback():

    return TensorBoard(

        log_dir=os.path.join(

            LOG_DIR,

            "tensorboard"

        ),

        histogram_freq=1,

        write_graph=True,

        write_images=True
    )


##############################################################
# Return All Callbacks
##############################################################

def get_callbacks():

    callbacks = [

        checkpoint_callback(),

        early_stopping_callback(),

        learning_rate_callback(),

        csv_logger_callback(),

        tensorboard_callback()

    ]

    return callbacks
