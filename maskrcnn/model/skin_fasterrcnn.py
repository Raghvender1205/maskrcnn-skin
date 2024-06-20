import json
import os
import random
import numpy as np
import tensorflow as tf
from typing import Any, Union
from loguru import logger

from maskrcnn.configs.fasterrcnn_config import FasterRCNNConfig
from maskrcnn.components import Backbone, RPN, RoI
from maskrcnn.utils import DataGenerator, BoundingBoxTools


class SkinConfig(FasterRCNNConfig):
    NAME = "skin_fasterrcnn"
    N_OUT_CLASS = 7  # 6 + 1(background)
    DATA_JSON_FILE = "D:/Internship/HealthKart/SkinSegmentation/HKSkin/custom_data/train/annotations.json"
    PATH_IMAGES = "D:/Internship/HealthKart/SkinSegmentation/HKSkin/custom_data/train/images"
    PATH_DEBUG_IMG = "D:/Internship/HealthKart/SkinSegmentation/HKSkin/custom_data/train/images/29900.png"

class FasterRCNNModel():
    def __init__(self):
        self.backbone = Backbone(img_shape=SkinConfig.IMG_RESIZED_SHAPE, n_stage=SkinConfig.N_STAGE)
        self.img_shape = SkinConfig.IMG_RESIZED_SHAPE
        # self.backbone.trainable = False

        # RPN
        self.rpn = RPN(backbone_model=self.backbone.backbone_model,
                       lambda_factor=SkinConfig.RPN_LAMBDA_FACTOR,
                       batch=SkinConfig.BATCH_RPN,
                       learning_rate=SkinConfig.LR)

        # ROI
        self.roi = RoI(backbone_model=self.backbone.backbone_model,
                       img_shape=self.img_shape,
                       n_output_classes=SkinConfig.N_OUT_CLASS,
                       learning_rate=SkinConfig.LR)
        self.roi_header = self.roi.roi_header_model

        # Data Generator
        self.train_data_generator = DataGenerator(
            file=SkinConfig.DATA_JSON_FILE,
            image_folder_path=SkinConfig.PATH_IMAGES,
            anchor_base_size=SkinConfig.ANCHOR_BASE_SIZE,
            ratios=SkinConfig.ANCHOR_RATIOS,
            scales=SkinConfig.ANCHOR_SCALES,
            n_anchors=SkinConfig.N_ANCHORS,
            img_shape_resize=SkinConfig.IMG_RESIZED_SHAPE,
            n_stage=SkinConfig.N_STAGE,
            threshold_iou_rpn=SkinConfig.THRESHOLD_IOU_RPN,
            threshold_iou_roi=SkinConfig.THRESHOLD_IOU_ROI
        )
        self.cocotool = self.train_data_generator.dataset_coco

        self.anchor_candidate_generator = self.train_data_generator.generate_candidate_anchors
        self.anchor_candidates = self.anchor_candidate_generator.anchor_candidates

    def nms_loop_np(self, boxes):
        # boxes : (N, 4), box_1target : (4,)
        # box axis format: (x1,y1,x2,y2)
        epsilon = 1e-6

        box_1target = np.ones(shape=boxes.shape)
        zeros = np.zeros(shape=boxes.shape)
        box_1target = box_1target * boxes[0, :]
        box_b_area = (box_1target[:, 2] - box_1target[:, 0] + 1) * (box_1target[:, 3] - box_1target[:, 1] + 1)

        # Determine the (x, y)-coordinates of the intersection rectangle ---
        x_a = np.max(np.array([boxes[:, 0], box_1target[:, 0]]), axis=0)
        y_a = np.max(np.array([boxes[:, 1], box_1target[:, 1]]), axis=0)
        x_b = np.min(np.array([boxes[:, 2], box_1target[:, 2]]), axis=0)
        y_b = np.min(np.array([boxes[:, 3], box_1target[:, 3]]), axis=0)

        # Compute the area of intersection rectangle ---
        inter_area = np.maximum(0, x_b - x_a + 1) * np.maximum(0, y_b - y_a + 1)
        # Compute area of both prediction and ground truth rectangles
        box_a_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        ious = inter_area / (box_a_area + box_b_area - inter_area + epsilon)

        return ious