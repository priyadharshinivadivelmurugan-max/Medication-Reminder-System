"""
=========================================================
CNN + BiLSTM + CTC OCR Inference
=========================================================
"""

import cv2
import numpy as np
import tensorflow as tf

from ocr.preprocess import ImagePreprocessor
from ocr.craft_detector import craft_detector
from ocr.cnn_bilstm_ctc import build_model
from ocr.decoder import decoder
from ocr.config import OCR_MODEL


class OCRInference:

    def __init__(self):

        self.preprocessor = ImagePreprocessor()

        self.model = self.load_model()

    ##########################################################
    # Load OCR Model
    ##########################################################

    def load_model(self):

        model = build_model()

        model.load_weights(OCR_MODEL)

        return model

    ##########################################################
    # Recognize One Word
    ##########################################################

    def recognize(self, crop):

        tensor = self.preprocessor.process_crop(crop)

        prediction = self.model.predict(

            tensor,

            verbose=0

        )

        prediction = prediction[0]

        text, confidence = decoder.decode(

            prediction

        )

        return text, confidence

    ##########################################################
    # OCR Pipeline
    ##########################################################

    def extract_text(self, image_path):

        image = cv2.imread(image_path)

        if image is None:

            raise ValueError(

                "Unable to read image."

            )

        regions = craft_detector.run(image)

        results = []

        confidences = []

        for region in regions:

            crop = region["image"]

            text, conf = self.recognize(crop)

            if len(text.strip()) == 0:

                continue

            results.append(text)

            confidences.append(conf)

        final_text = " ".join(results)

        if len(confidences):

            avg_conf = np.mean(confidences)

        else:

            avg_conf = 0.0

        return final_text, round(float(avg_conf),4)

    ##########################################################
    # OCR with Bounding Boxes
    ##########################################################

    def extract_with_boxes(self, image_path):

        image = cv2.imread(image_path)

        regions = craft_detector.run(image)

        predictions = []

        for region in regions:

            crop = region["image"]

            box = region["box"]

            text, conf = self.recognize(crop)

            predictions.append({

                "text": text,

                "confidence": conf,

                "box": box.tolist()

            })

        return predictions


##############################################################
# Singleton
##############################################################

ocr_engine = OCRInference()


##############################################################
# Compatible Function
##############################################################

def cnn_bilstm_ctc_ocr(image_path):

    return ocr_engine.extract_text(

        image_path

    )


##############################################################
# Test
##############################################################

if __name__ == "__main__":

    image = "sample.jpg"

    text, conf = cnn_bilstm_ctc_ocr(image)

    print()

    print("OCR TEXT")

    print("-------------------------")

    print(text)

    print()

    print("CONFIDENCE")

    print(conf)
