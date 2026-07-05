"""
=========================================================
Train CNN + BiLSTM + CTC OCR Model
=========================================================
"""

import os
import tensorflow as tf

from training.dataloader import OCRDataLoader
from training.model import build_training_model
from training.loss import compile_model, print_training_configuration
from training.callbacks import get_callbacks

from ocr.config import *


##############################################################
# GPU Configuration
##############################################################

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:

        print(e)


##############################################################
# Dataset Paths
##############################################################

TRAIN_IMAGE_FOLDER = os.path.join(
    DATASET_DIR,
    "train"
)

TRAIN_LABEL_FILE = os.path.join(
    DATASET_DIR,
    "labels.csv"
)


##############################################################
# Data Loader
##############################################################

print("\nLoading Dataset...\n")

train_loader = OCRDataLoader(

    image_folder=TRAIN_IMAGE_FOLDER,

    label_file=TRAIN_LABEL_FILE,

    batch_size=BATCH_SIZE,

    shuffle=True

)


##############################################################
# Build Model
##############################################################

print("Building CNN + BiLSTM + CTC Model...\n")

model = build_training_model()


##############################################################
# Compile Model
##############################################################

model = compile_model(model)

print_training_configuration()


##############################################################
# Train Model
##############################################################

print("\nStarting Training...\n")

history = model.fit(

    train_loader,

    validation_data=train_loader,

    epochs=EPOCHS,

    callbacks=get_callbacks(),

    verbose=1

)


##############################################################
# Save Final Model
##############################################################

os.makedirs(

    MODEL_DIR,

    exist_ok=True

)

FINAL_MODEL = os.path.join(

    MODEL_DIR,

    "ocr_model.keras"

)

model.save(

    FINAL_MODEL

)

print("\n====================================")

print("Training Completed Successfully")

print(f"Model Saved : {FINAL_MODEL}")

print("====================================")


##############################################################
# Save History
##############################################################

import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv(

    os.path.join(

        LOG_DIR,

        "history.csv"

    ),

    index=False

)

print("\nTraining history saved.")


##############################################################
# Evaluate
##############################################################

print("\nRun evaluate.py to evaluate the trained model.")
