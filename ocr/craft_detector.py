"""
=========================================================
CRAFT Text Detection Module
CNN + BiLSTM + CTC OCR Pipeline
=========================================================
"""

import cv2
import torch
import numpy as np

from craft import CRAFT
import craft_utils
import imgproc

from collections import OrderedDict

from ocr.config import *


class CraftDetector:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.load_model()

    ##########################################################
    # Load CRAFT Model
    ##########################################################

    def load_model(self):

        network = CRAFT()

        state_dict = torch.load(
            CRAFT_MODEL,
            map_location=self.device
        )

        new_state = OrderedDict()

        for key, value in state_dict.items():

            name = key.replace("module.", "")

            new_state[name] = value

        network.load_state_dict(new_state)

        network.to(self.device)

        network.eval()

        return network

    ##########################################################
    # Detect Text Regions
    ##########################################################

    def detect(self, image):

        image_resized, target_ratio, _ = imgproc.resize_aspect_ratio(

            image,

            CANVAS_SIZE,

            interpolation=cv2.INTER_LINEAR,

            mag_ratio=MAG_RATIO

        )

        ratio_h = ratio_w = 1 / target_ratio

        normalized = imgproc.normalizeMeanVariance(image_resized)

        tensor = torch.from_numpy(normalized)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        with torch.no_grad():

            prediction, _ = self.model(tensor)

        score_text = prediction[0, :, :, 0].cpu().numpy()

        score_link = prediction[0, :, :, 1].cpu().numpy()

        boxes, polys = craft_utils.getDetBoxes(

            score_text,

            score_link,

            TEXT_THRESHOLD,

            LINK_THRESHOLD,

            LOW_TEXT,

            False

        )

        boxes = craft_utils.adjustResultCoordinates(

            boxes,

            ratio_w,

            ratio_h

        )

        return boxes

    ##########################################################
    # Crop Text Regions
    ##########################################################

    def crop_regions(self, image, boxes):

        crops = []

        for box in boxes:

            points = np.array(box).astype(np.int32)

            x_min = np.min(points[:, 0])
            y_min = np.min(points[:, 1])

            x_max = np.max(points[:, 0])
            y_max = np.max(points[:, 1])

            crop = image[
                y_min:y_max,
                x_min:x_max
            ]

            if crop.size == 0:
                continue

            crops.append({

                "image": crop,

                "box": box

            })

        return crops

    ##########################################################
    # Complete Detection Pipeline
    ##########################################################

    def run(self, image):

        boxes = self.detect(image)

        crops = self.crop_regions(

            image,

            boxes

        )

        return crops


##############################################################
# Singleton Object
##############################################################

craft_detector = CraftDetector()
