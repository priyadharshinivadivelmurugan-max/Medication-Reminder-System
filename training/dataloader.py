"""
=========================================================
OCR DataLoader
CNN + BiLSTM + CTC
=========================================================
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from training.dataset import OCRDataset

from ocr.config import *


class OCRDataLoader(tf.keras.utils.Sequence):

    def __init__(
            self,
            image_folder,
            label_file,
            batch_size=BATCH_SIZE,
            shuffle=True):

        self.dataset = OCRDataset(
            image_folder,
            label_file
        )

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indexes = np.arange(
            len(self.dataset)
        )

        self.on_epoch_end()

    ##########################################################
    # Number of Batches
    ##########################################################

    def __len__(self):

        return int(

            np.ceil(

                len(self.dataset) /

                self.batch_size

            )

        )

    ##########################################################
    # One Batch
    ##########################################################

    def __getitem__(self, index):

        batch_indexes = self.indexes[

            index * self.batch_size:

            (index + 1) * self.batch_size

        ]

        images = []

        labels = []

        input_lengths = []

        label_lengths = []

        for idx in batch_indexes:

            sample = self.dataset[idx]

            images.append(

                sample["image"]

            )

            labels.append(

                sample["encoded"]

            )

            input_lengths.append(

                sample["input_length"]

            )

            label_lengths.append(

                sample["label_length"]

            )

        images = np.array(

            images,

            dtype=np.float32

        )

        labels = pad_sequences(

            labels,

            padding="post",

            value=0

        )

        input_lengths = np.array(

            input_lengths

        ).reshape(-1,1)

        label_lengths = np.array(

            label_lengths

        ).reshape(-1,1)

        inputs = {

            "image": images,

            "labels": labels,

            "input_length": input_lengths,

            "label_length": label_lengths

        }

        outputs = np.zeros(

            len(images)

        )

        return inputs, outputs

    ##########################################################
    # Shuffle Every Epoch
    ##########################################################

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(

                self.indexes

            )
