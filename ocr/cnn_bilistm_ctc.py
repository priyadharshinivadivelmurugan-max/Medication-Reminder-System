"""
=========================================================
CNN + BiLSTM + CTC Recognition Model
=========================================================
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Reshape,
    Dense,
    Bidirectional,
    LSTM,
    Dropout
)

from tensorflow.keras.models import Model

from ocr.config import *
from ocr.vocabulary import vocabulary


class CRNN:

    def __init__(self):

        self.image_height = IMAGE_HEIGHT
        self.image_width = IMAGE_WIDTH
        self.channels = CHANNELS

        self.num_classes = vocabulary.size()

    ##########################################################
    # CNN Feature Extractor
    ##########################################################

    def cnn_backbone(self, inputs):

        x = Conv2D(
            64,
            (3,3),
            padding="same"
        )(inputs)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2,2))(x)

        x = Conv2D(
            128,
            (3,3),
            padding="same"
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2,2))(x)

        x = Conv2D(
            256,
            (3,3),
            padding="same"
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(
            256,
            (3,3),
            padding="same"
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D(
            pool_size=(2,1)
        )(x)

        x = Conv2D(
            512,
            (3,3),
            padding="same"
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(
            512,
            (3,3),
            padding="same"
        )(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D(
            pool_size=(2,1)
        )(x)

        return x

    ##########################################################
    # Convert CNN Feature Map to Sequence
    ##########################################################

    def sequence_layer(self, x):

        shape = x.shape

        x = Reshape(
            target_shape=(
                shape[2],
                shape[1] * shape[3]
            )
        )(x)

        return x

    ##########################################################
    # BiLSTM Layers
    ##########################################################

    def recurrent_layers(self, x):

        x = Bidirectional(

            LSTM(

                256,

                return_sequences=True

            )

        )(x)

        x = Dropout(0.25)(x)

        x = Bidirectional(

            LSTM(

                256,

                return_sequences=True

            )

        )(x)

        x = Dropout(0.25)(x)

        return x

    ##########################################################
    # Character Prediction
    ##########################################################

    def prediction_layer(self, x):

        outputs = Dense(

            self.num_classes,

            activation="softmax",

            name="character_probabilities"

        )(x)

        return outputs

    ##########################################################
    # Build Complete Model
    ##########################################################

    def build(self):

        inputs = Input(

            shape=(

                self.image_height,

                self.image_width,

                self.channels

            ),

            name="input_image"

        )

        x = self.cnn_backbone(inputs)

        x = self.sequence_layer(x)

        x = self.recurrent_layers(x)

        outputs = self.prediction_layer(x)

        model = Model(

            inputs=inputs,

            outputs=outputs,

            name="CNN_BiLSTM_CTC"

        )

        return model


##############################################################
# Helper Function
##############################################################

def build_model():

    network = CRNN()

    return network.build()


##############################################################
# Test
##############################################################

if __name__ == "__main__":

    model = build_model()

    model.summary()
