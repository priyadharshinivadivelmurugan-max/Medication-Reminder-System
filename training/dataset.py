"""
=========================================================
Dataset Loader
CNN + BiLSTM + CTC OCR
=========================================================
"""

import os
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split

from ocr.preprocess import ImagePreprocessor
from ocr.vocabulary import vocabulary


class OCRDataset:

    def __init__(self,
                 image_folder,
                 label_file):

        self.image_folder = image_folder
        self.label_file = label_file

        self.preprocessor = ImagePreprocessor()

        self.data = self.load_annotations()

    ############################################################
    # Load CSV Labels
    ############################################################

    def load_annotations(self):

        if not os.path.exists(self.label_file):

            raise FileNotFoundError(
                f"{self.label_file} not found."
            )

        df = pd.read_csv(self.label_file)

        return df

    ############################################################
    # Dataset Size
    ############################################################

    def __len__(self):

        return len(self.data)

    ############################################################
    # Read Image
    ############################################################

    def load_image(self, filename):

        path = os.path.join(
            self.image_folder,
            filename
        )

        image = cv2.imread(path)

        if image is None:

            raise ValueError(
                f"Unable to read {path}"
            )

        return image

    ############################################################
    # Preprocess Image
    ############################################################

    def preprocess(self, image):

        tensor = self.preprocessor.process_crop(
            image
        )

        return tensor[0]

    ############################################################
    # Encode Label
    ############################################################

    def encode_label(self, text):

        return vocabulary.encode(text)

    ############################################################
    # Get One Sample
    ############################################################

    def __getitem__(self, index):

        row = self.data.iloc[index]

        filename = row["filename"]

        label = row["label"]

        image = self.load_image(filename)

        image = self.preprocess(image)

        label_encoded = self.encode_label(label)

        sample = {

            "image": image,

            "label": label,

            "encoded": label_encoded,

            "label_length": len(label_encoded),

            "input_length": image.shape[1] // 4

        }

        return sample

    ############################################################
    # Train Validation Split
    ############################################################

    def split(self,
              validation_size=0.2,
              random_state=42):

        train_df, val_df = train_test_split(

            self.data,

            test_size=validation_size,

            random_state=random_state,

            shuffle=True

        )

        return train_df, val_df

    ############################################################
    # Character Statistics
    ############################################################

    def character_statistics(self):

        stats = {}

        for label in self.data["label"]:

            for c in label:

                stats[c] = stats.get(c, 0) + 1

        return stats

    ############################################################
    # Maximum Label Length
    ############################################################

    def max_label_length(self):

        return max(

            len(str(x))

            for x in self.data["label"]

        )

    ############################################################
    # Vocabulary Coverage
    ############################################################

    def unknown_characters(self):

        unknown = set()

        for label in self.data["label"]:

            for c in label:

                if not vocabulary.contains(c):

                    unknown.add(c)

        return sorted(list(unknown))
