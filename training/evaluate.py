"""
=========================================================
Model Evaluation
CNN + BiLSTM + CTC OCR
=========================================================
"""

import numpy as np
from difflib import SequenceMatcher
from tensorflow.keras.models import load_model

from training.dataset import OCRDataset
from ocr.preprocess import ImagePreprocessor
from ocr.decoder import decoder
from ocr.config import OCR_MODEL


class OCREvaluator:

    def __init__(self,
                 image_folder,
                 label_file):

        self.dataset = OCRDataset(
            image_folder,
            label_file
        )

        self.preprocessor = ImagePreprocessor()

        self.model = self.load_model()

    ##########################################################
    # Load Model
    ##########################################################

    def load_model(self):

        model = load_model(
            OCR_MODEL,
            compile=False
        )

        return model

    ##########################################################
    # Predict One Sample
    ##########################################################

    def predict(self, image):

        tensor = self.preprocessor.process_crop(image)

        prediction = self.model.predict(
            tensor,
            verbose=0
        )

        prediction = prediction[0]

        text, confidence = decoder.decode(prediction)

        return text, confidence

    ##########################################################
    # Character Accuracy
    ##########################################################

    def character_accuracy(
            self,
            truth,
            prediction):

        return SequenceMatcher(
            None,
            truth,
            prediction
        ).ratio()

    ##########################################################
    # Word Accuracy
    ##########################################################

    def word_accuracy(
            self,
            truth,
            prediction):

        return truth.strip().lower() == prediction.strip().lower()

    ##########################################################
    # Evaluate Entire Dataset
    ##########################################################

    def evaluate(self):

        total = len(self.dataset)

        correct = 0

        character_scores = []

        confidence_scores = []

        print("\n======================================")

        print("OCR Evaluation")

        print("======================================")

        for i in range(total):

            sample = self.dataset[i]

            image = sample["image"]

            truth = sample["label"]

            prediction, confidence = self.predict(image)

            char_acc = self.character_accuracy(
                truth,
                prediction
            )

            if self.word_accuracy(
                    truth,
                    prediction):

                correct += 1

            character_scores.append(
                char_acc
            )

            confidence_scores.append(
                confidence
            )

            print(f"\nSample : {i+1}")

            print(f"Ground Truth : {truth}")

            print(f"Prediction   : {prediction}")

            print(f"Confidence   : {confidence:.4f}")

            print(f"Char Accuracy: {char_acc:.4f}")

        print("\n======================================")

        print("FINAL RESULTS")

        print("======================================")

        print(f"Word Accuracy : {(correct/total)*100:.2f}%")

        print(f"Character Accuracy : {np.mean(character_scores)*100:.2f}%")

        print(f"Average Confidence : {np.mean(confidence_scores):.4f}")

        print("======================================")
