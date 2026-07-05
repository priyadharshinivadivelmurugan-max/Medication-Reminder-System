"""
=========================================================
CTC Decoder
CNN + BiLSTM + CTC OCR
=========================================================
"""

import numpy as np
import tensorflow as tf

from ocr.vocabulary import vocabulary


class CTCDecoder:

    def __init__(self):

        self.blank = vocabulary.blank_token

    ##########################################################
    # Greedy Decoder
    ##########################################################

    def greedy_decode(self, prediction):

        sequence = np.argmax(prediction, axis=-1)

        text = ""
        previous = self.blank

        for index in sequence:

            index = int(index)

            if index == self.blank:
                previous = index
                continue

            if index == previous:
                continue

            character = vocabulary.get_character(index)

            if character != "<BLANK>":
                text += character

            previous = index

        return text

    ##########################################################
    # TensorFlow CTC Decoder
    ##########################################################

    def tensorflow_decode(self, prediction):

        prediction = np.expand_dims(prediction, axis=0)

        input_length = np.ones(
            prediction.shape[0]
        ) * prediction.shape[1]

        decoded, _ = tf.keras.backend.ctc_decode(

            prediction,

            input_length=input_length,

            greedy=True

        )

        decoded = decoded[0].numpy()[0]

        text = ""

        for idx in decoded:

            if idx == -1:
                continue

            text += vocabulary.get_character(int(idx))

        return text

    ##########################################################
    # Confidence Score
    ##########################################################

    def confidence(self, prediction):

        probabilities = np.max(
            prediction,
            axis=-1
        )

        confidence = float(

            np.mean(probabilities)

        )

        return round(confidence, 4)

    ##########################################################
    # Decode Prediction
    ##########################################################

    def decode(self, prediction):

        text = self.greedy_decode(prediction)

        score = self.confidence(prediction)

        return text, score

    ##########################################################
    # Beam Search (Placeholder)
    ##########################################################

    def beam_search(self, prediction):

        prediction = np.expand_dims(
            prediction,
            axis=0
        )

        input_length = np.ones(

            prediction.shape[0]

        ) * prediction.shape[1]

        decoded, _ = tf.keras.backend.ctc_decode(

            prediction,

            input_length,

            greedy=False,

            beam_width=10,

            top_paths=1

        )

        decoded = decoded[0].numpy()[0]

        text = ""

        for idx in decoded:

            if idx == -1:
                continue

            text += vocabulary.get_character(

                int(idx)

            )

        confidence = self.confidence(prediction[0])

        return text, confidence


##############################################################
# Singleton Object
##############################################################

decoder = CTCDecoder()
