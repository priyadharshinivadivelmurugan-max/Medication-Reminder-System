"""
=========================================================
Vocabulary Module
CNN + BiLSTM + CTC OCR
=========================================================
"""

import os

from ocr.config import VOCAB_FILE


class Vocabulary:

    def __init__(self):

        self.characters = []

        self.char_to_index = {}

        self.index_to_char = {}

        self.blank_token = 0

        self.load()


    ##########################################################
    # Load Vocabulary
    ##########################################################

    def load(self):

        if not os.path.exists(VOCAB_FILE):

            raise FileNotFoundError(
                f"Vocabulary file not found : {VOCAB_FILE}"
            )

        with open(
            VOCAB_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            chars = [line.strip() for line in f.readlines()]

        # CTC Blank Label

        self.characters = ["<BLANK>"] + chars

        self.char_to_index = {

            c: i

            for i, c in enumerate(self.characters)

        }

        self.index_to_char = {

            i: c

            for i, c in enumerate(self.characters)

        }


    ##########################################################
    # Number of Classes
    ##########################################################

    def size(self):

        return len(self.characters)


    ##########################################################
    # Encode Text
    ##########################################################

    def encode(self, text):

        sequence = []

        for ch in text:

            if ch in self.char_to_index:

                sequence.append(

                    self.char_to_index[ch]

                )

        return sequence


    ##########################################################
    # Decode Integer Sequence
    ##########################################################

    def decode(self, sequence):

        text = ""

        previous = -1

        for idx in sequence:

            idx = int(idx)

            # Skip Blank

            if idx == self.blank_token:

                previous = idx

                continue

            # Remove consecutive duplicates

            if idx == previous:

                continue

            previous = idx

            text += self.index_to_char.get(idx, "")

        return text


    ##########################################################
    # Decode Prediction
    ##########################################################

    def decode_prediction(self, prediction):

        import numpy as np

        indices = np.argmax(

            prediction,

            axis=-1

        )

        return self.decode(indices)


    ##########################################################
    # Check Character Exists
    ##########################################################

    def contains(self, character):

        return character in self.char_to_index


    ##########################################################
    # Get Character Index
    ##########################################################

    def get_index(self, character):

        return self.char_to_index.get(

            character,

            self.blank_token

        )


    ##########################################################
    # Get Character
    ##########################################################

    def get_character(self, index):

        return self.index_to_char.get(

            index,

            ""

        )


##############################################################
# Singleton Object
##############################################################

vocabulary = Vocabulary()
