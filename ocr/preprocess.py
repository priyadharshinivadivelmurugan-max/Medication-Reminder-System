"""
=========================================================
Image Preprocessing Module
CNN + BiLSTM + CTC OCR Pipeline
=========================================================
"""

import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(
            self,
            img_height=32,
            img_width=128):

        self.height = img_height
        self.width = img_width

    ############################################################
    # Read image
    ############################################################

    def read_image(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Cannot load image : {image_path}")

        return img

    ############################################################
    # Convert to grayscale
    ############################################################

    def grayscale(self, image):

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ############################################################
    # CLAHE enhancement
    ############################################################

    def enhance_contrast(self, gray):

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(gray)

    ############################################################
    # Noise removal
    ############################################################

    def denoise(self, gray):

        return cv2.fastNlMeansDenoising(
            gray,
            None,
            10,
            7,
            21
        )

    ############################################################
    # Adaptive threshold
    ############################################################

    def threshold(self, gray):

        return cv2.adaptiveThreshold(

            gray,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY,

            31,

            15
        )

    ############################################################
    # Resize while preserving aspect ratio
    ############################################################

    def resize_keep_ratio(self, image):

        h, w = image.shape

        scale = min(
            self.width / w,
            self.height / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            image,
            (new_w, new_h)
        )

        canvas = np.ones(
            (self.height, self.width),
            dtype=np.uint8
        ) * 255

        canvas[
            0:new_h,
            0:new_w
        ] = resized

        return canvas

    ############################################################
    # Normalize image
    ############################################################

    def normalize(self, image):

        image = image.astype(np.float32)

        image /= 255.0

        return image

    ############################################################
    # Expand dimension for CNN
    ############################################################

    def prepare_tensor(self, image):

        image = np.expand_dims(image, axis=-1)

        image = np.expand_dims(image, axis=0)

        return image

    ############################################################
    # Complete preprocessing pipeline
    ############################################################

    def process(self, image_path):

        image = self.read_image(image_path)

        gray = self.grayscale(image)

        gray = self.enhance_contrast(gray)

        gray = self.denoise(gray)

        binary = self.threshold(gray)

        resized = self.resize_keep_ratio(binary)

        normalized = self.normalize(resized)

        tensor = self.prepare_tensor(normalized)

        return tensor

    ############################################################
    # Process cropped text image
    ############################################################

    def process_crop(self, crop):

        gray = self.grayscale(crop)

        gray = self.enhance_contrast(gray)

        gray = self.denoise(gray)

        binary = self.threshold(gray)

        resized = self.resize_keep_ratio(binary)

        normalized = self.normalize(resized)

        tensor = self.prepare_tensor(normalized)

        return tensor
