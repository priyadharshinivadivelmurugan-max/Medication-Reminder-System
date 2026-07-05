"""
=========================================================
Training Model
CNN + BiLSTM + CTC OCR
=========================================================
"""

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Lambda
)

from tensorflow.keras import backend as K

from ocr.cnn_bilstm_ctc import build_model


##############################################################
# CTC LOSS FUNCTION
##############################################################

def ctc_lambda(args):

    y_pred, labels, input_length, label_length = args

    loss = K.ctc_batch_cost(

        labels,

        y_pred,

        input_length,

        label_length

    )

    return loss


##############################################################
# BUILD TRAINING MODEL
##############################################################

def build_training_model():

    base_model = build_model()

    labels = Input(

        name="labels",

        shape=(None,),

        dtype="float32"

    )

    input_length = Input(

        name="input_length",

        shape=(1,),

        dtype="int64"

    )

    label_length = Input(

        name="label_length",

        shape=(1,),

        dtype="int64"

    )

    loss_output = Lambda(

        ctc_lambda,

        output_shape=(1,),

        name="ctc_loss"

    )([

        base_model.output,

        labels,

        input_length,

        label_length

    ])

    training_model = Model(

        inputs=[

            base_model.input,

            labels,

            input_length,

            label_length

        ],

        outputs=loss_output,

        name="OCR_CTC_Training_Model"

    )

    return training_model


##############################################################
# TEST
##############################################################

if __name__ == "__main__":

    model = build_training_model()

    model.summary()
