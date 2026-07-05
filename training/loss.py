"""
=========================================================
Loss & Optimizer Configuration
CNN + BiLSTM + CTC OCR
=========================================================
"""

import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from ocr.config import LEARNING_RATE


##############################################################
# Dummy Loss
##############################################################

def dummy_ctc_loss(y_true, y_pred):

    """
    Since the CTC loss is already computed inside the
    Lambda layer (ctc_loss), Keras still expects a loss
    function during model compilation.

    Therefore, this function simply returns the predicted
    loss value.
    """

    return y_pred


##############################################################
# Optimizer
##############################################################

def get_optimizer():

    optimizer = Adam(

        learning_rate=LEARNING_RATE,

        beta_1=0.9,

        beta_2=0.999,

        epsilon=1e-07

    )

    return optimizer


##############################################################
# Learning Rate Scheduler
##############################################################

def learning_rate_scheduler():

    scheduler = tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        verbose=1,

        min_lr=1e-6

    )

    return scheduler


##############################################################
# Model Compilation
##############################################################

def compile_model(model):

    model.compile(

        optimizer=get_optimizer(),

        loss=dummy_ctc_loss

    )

    return model


##############################################################
# Print Configuration
##############################################################

def print_training_configuration():

    print("\n==============================")

    print("Training Configuration")

    print("==============================")

    print(f"Optimizer      : Adam")

    print(f"Learning Rate  : {LEARNING_RATE}")

    print("Loss Function  : CTC Loss")

    print("==============================\n")
