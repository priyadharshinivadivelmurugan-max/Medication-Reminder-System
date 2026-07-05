"""
=========================================================
OCR Configuration File
CNN + BiLSTM + CTC
=========================================================
"""

import os


############################################################
# PROJECT PATHS
############################################################

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OCR_DIR = os.path.join(ROOT_DIR, "ocr")

MODEL_DIR = os.path.join(OCR_DIR, "models")

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

OCR_RESULTS_DIR = os.path.join(OUTPUT_DIR, "ocr_results")


############################################################
# MODEL PATHS
############################################################

CRAFT_MODEL = os.path.join(
    MODEL_DIR,
    "craft_mlt_25k.pth"
)

OCR_MODEL = os.path.join(
    MODEL_DIR,
    "ocr_model.keras"
)

VOCAB_FILE = os.path.join(
    DATASET_DIR,
    "vocabulary.txt"
)


############################################################
# IMAGE SETTINGS
############################################################

IMAGE_HEIGHT = 32

IMAGE_WIDTH = 128

CHANNELS = 1


############################################################
# CRAFT PARAMETERS
############################################################

TEXT_THRESHOLD = 0.7

LINK_THRESHOLD = 0.4

LOW_TEXT = 0.4

CANVAS_SIZE = 1280

MAG_RATIO = 1.5


############################################################
# CNN PARAMETERS
############################################################

CNN_FILTERS = [64, 128, 256, 512]

KERNEL_SIZE = (3, 3)

POOL_SIZE = (2, 2)

DROPOUT = 0.25


############################################################
# BiLSTM PARAMETERS
############################################################

LSTM_UNITS = 256

NUM_LSTM_LAYERS = 2


############################################################
# TRAINING PARAMETERS
############################################################

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.0001

VALIDATION_SPLIT = 0.2


############################################################
# CTC PARAMETERS
############################################################

MAX_LABEL_LENGTH = 50

BLANK_LABEL = 0


############################################################
# DECODER
############################################################

DECODER_TYPE = "greedy"

# Options:
# greedy
# beam_search


############################################################
# OUTPUT
############################################################

SAVE_PREDICTIONS = True

SAVE_BOUNDING_BOXES = True

SAVE_CONFIDENCE = True


############################################################
# LOGGING
############################################################

VERBOSE = True

SHOW_PROCESS = True
