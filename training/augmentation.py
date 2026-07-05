"""
=========================================================
Data Augmentation
CNN + BiLSTM + CTC OCR
=========================================================
"""

import cv2
import numpy as np
import random


class OCRAugmentation:

    def __init__(self):

        pass

    ##########################################################
    # Random Rotation
    ##########################################################

    def random_rotation(self, image):

        angle = random.uniform(-5, 5)

        h, w = image.shape[:2]

        matrix = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            1.0
        )

        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

    ##########################################################
    # Random Brightness
    ##########################################################

    def random_brightness(self, image):

        alpha = random.uniform(0.8, 1.2)

        beta = random.randint(-20, 20)

        image = cv2.convertScaleAbs(

            image,

            alpha=alpha,

            beta=beta

        )

        return image

    ##########################################################
    # Gaussian Blur
    ##########################################################

    def gaussian_blur(self, image):

        if random.random() < 0.4:

            image = cv2.GaussianBlur(

                image,

                (3,3),

                0

            )

        return image

    ##########################################################
    # Add Gaussian Noise
    ##########################################################

    def gaussian_noise(self, image):

        if random.random() < 0.4:

            noise = np.random.normal(

                0,

                10,

                image.shape

            )

            image = image + noise

            image = np.clip(

                image,

                0,

                255

            ).astype(np.uint8)

        return image

    ##########################################################
    # Horizontal Shift
    ##########################################################

    def horizontal_shift(self, image):

        h, w = image.shape[:2]

        shift = random.randint(-5,5)

        matrix = np.float32(

            [

                [1,0,shift],

                [0,1,0]

            ]

        )

        return cv2.warpAffine(

            image,

            matrix,

            (w,h)

        )

    ##########################################################
    # Vertical Shift
    ##########################################################

    def vertical_shift(self, image):

        h, w = image.shape[:2]

        shift = random.randint(-3,3)

        matrix = np.float32(

            [

                [1,0,0],

                [0,1,shift]

            ]

        )

        return cv2.warpAffine(

            image,

            matrix,

            (w,h)

        )

    ##########################################################
    # Random Scaling
    ##########################################################

    def random_scale(self, image):

        scale = random.uniform(

            0.9,

            1.1

        )

        h, w = image.shape[:2]

        image = cv2.resize(

            image,

            None,

            fx=scale,

            fy=scale

        )

        image = cv2.resize(

            image,

            (w,h)

        )

        return image

    ##########################################################
    # Morphological Opening
    ##########################################################

    def opening(self, image):

        kernel = np.ones(

            (2,2),

            np.uint8

        )

        return cv2.morphologyEx(

            image,

            cv2.MORPH_OPEN,

            kernel

        )

    ##########################################################
    # Morphological Closing
    ##########################################################

    def closing(self, image):

        kernel = np.ones(

            (2,2),

            np.uint8

        )

        return cv2.morphologyEx(

            image,

            cv2.MORPH_CLOSE,

            kernel

        )

    ##########################################################
    # Complete Pipeline
    ##########################################################

    def augment(self, image):

        image = self.random_rotation(image)

        image = self.random_brightness(image)

        image = self.gaussian_blur(image)

        image = self.gaussian_noise(image)

        image = self.horizontal_shift(image)

        image = self.vertical_shift(image)

        image = self.random_scale(image)

        image = self.opening(image)

        image = self.closing(image)

        return image


##############################################################
# Singleton
##############################################################

augmentor = OCRAugmentation()
