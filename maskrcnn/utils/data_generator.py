import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

from loguru import logger
from maskrcnn.utils import BoundingBoxTools, GenerateCandidateAnchors, COCOTools


class DataGenerator:
    def __init__(self, file: str, image_folder_path: str,
                 anchor_base_size: int, ratios: list,
                 scales, n_anchors: int,
                 img_shape_resize: tuple = (512, 512, 3),
                 n_stage: int = 5,
                 threshold_iou_rpn: float = 0.7, threshold_iou_roi: float = 0.55):
        self.threshold_iou_rpn = threshold_iou_rpn
        self.threshold_iou_roi = threshold_iou_roi
        self.dataset_coco = COCOTools(file, image_folder_path, img_shape_resize)
        self.generate_candidate_anchors = GenerateCandidateAnchors(base_size=anchor_base_size, ratios=ratios,
                                                                   scales=scales,
                                                                   img_shape=img_shape_resize, n_stage=n_stage,
                                                                   n_anchors=n_anchors)
        self.img_shape_resize = img_shape_resize

    def _resize_img(self, img):
        return cv2.resize(img, self.img_shape_resize, interpolation=cv2.INTER_LINEAR)

    def _resize_box(self, box):
        pass

    def generate_train_input_one(self, image_id):
        return self.dataset_coco.get_original_image(image_id=image_id)

    def generate_train_target_box_regression_for_rpn(self, image_id, debuginfo=False):
        bboxes = self.dataset_coco.get_original_bboxes_list(image_id=image_id)

        # For each gt_bbox calculate ious with candidates
        bboxes_ious = []
        for bbox in bboxes:
            ious = BoundingBoxTools.ious(self.generate_candidate_anchors.anchor_candidates_list, bbox)
            ious_temp = np.ones(shape=(len(ious)), dtype=np.float32) * 0.5  # 0.5 is to indicate ignoring
            ious_temp = np.where(np.asarray(ious) > self.threshold_iou_rpn, 1, ious_temp)
            ious_temp = np.where(np.asarray(ious) < 0.3, 0, ious_temp)
            ious_temp[np.argmax(ious)] = 1
            bboxes_ious.append(ious_temp)

        # For each candidate anchor, determine the anchor target
        anchors_target = np.array(bboxes_ious, dtype=np.float32)
        anchors_target = np.max(anchors_target, axis=0)
        anchors_target = np.reshape(anchors_target, (self.generate_candidate_anchors.h, self.generate_candidate_anchors.w, self.generate_candidate_anchors.n_anchors))
        if debuginfo:
            logger.debug(f"Number of total gt bboxes :{len(bboxes)}")
            logger.debug(f"Number of total target anchors: {anchors_target[np.where(anchors_target == 1)].shape[0]}")
            logger.debug(f"Shape of anchors_target: {anchors_target.shape}")
            logger.debug(f"Selected Anchors: \n {self.generate_candidate_anchors.anchor_candidates[np.where(anchors_target == 1)]}")

        # For each gt_box, determine the box regression target
        bbox_reg_target = np.zeros(
            shape=(self.generate_candidate_anchors.h, self.generate_candidate_anchors.w, self.generate_candidate_anchors.n_anchors, 4),
            dtype=np.float32)
        for idx, bbox_ious in enumerate(bboxes_ious):
            ious_temp = np.reshape(bbox_ious, newshape=(
                self.generate_candidate_anchors.h, self.generate_candidate_anchors.w, self.generate_candidate_anchors.n_anchors))
            gt_box = bboxes[idx]
            candidate_boxes = self.generate_candidate_anchors.anchor_candidates[np.where(ious_temp == 1)]
            box_reg = BoundingBoxTools.bbox_regression_target(candidate_boxes, gt_box)
            logger.info(box_reg)
            logger.debug(BoundingBoxTools.bbox_reg2truebox(candidate_boxes, box_reg))
            bbox_reg_target[np.where(ious_temp == 1)] = box_reg

        return anchors_target, bbox_reg_target

    def generate_target_anchor_bboxes_classes_for_debug(self, image_id, debuginfo=False):
        bboxes = self.dataset_coco.get_original_bboxes_list(image_id=image_id)
        sparse_targets = self.dataset_coco.get_original_category_sparse_list(image_id=image_id)

        bboxes_ious = []
        for bbox in bboxes:
            ious = BoundingBoxTools.ious(self.generate_candidate_anchors.anchor_candidates_list, bbox)
            ious_temp = np.ones(shape=(len(ious)), dtype=np.float32) * 0.5
            # 0.5 to use max
            ious_temp = np.where(np.asarray(ious) > self.threshold_iou_rpn, 1, ious_temp)
            ious_temp = np.where(np.asarray(ious) < 0.3, 0, ious_temp)
            ious_temp[np.argmax(ious)] = 1
            bboxes_ious.append(ious_temp)

        # For each gt_box, determine box reg target
        target_anchor_bboxes = []
        target_classes = []
        for idx, bbox_ious in enumerate(bboxes_ious):
            ious_temp = np.reshape(bbox_ious, newshape=(
                self.generate_candidate_anchors.h, self.generate_candidate_anchors.w, self.generate_candidate_anchors.n_anchors
            ))
            candidate_boxes = self.generate_candidate_anchors.anchor_candidates[np.where(ious_temp == 1)]
            n = candidate_boxes.shape[0]
            for i in range(n):
                target_anchor_bboxes.append(candidate_boxes[i])
                target_classes.append(sparse_targets[idx])

        return target_anchor_bboxes, target_classes

    def generate_train_data_rpn_one(self, image_id):
        input1 = self.generate_train_input_one(image_id)
        anchor_target, bbox_reg_target = self.generate_train_target_box_regression_for_rpn(image_id)

        return np.array([input1]).astype(np.float32), np.array([anchor_target]).astype(np.float32), np.array(
            [bbox_reg_target]).astype(np.float32)

    def generate_train_data_rpn_all(self):
        inputs = []
        anchor_targets = []
        bbox_reg_targets = []
        for image_id in self.dataset_coco.image_ids:
            inputs.append(self.generate_train_input_one(image_id))
            anchor_target, bbox_reg_target = self.generate_train_target_box_regression_for_rpn(image_id)
            anchor_targets.append(anchor_target)
            bbox_reg_targets.append(bbox_reg_target)

        return np.array(inputs), np.array(anchor_targets), np.array(bbox_reg_targets)

    def generate_train_data_roi_one(self, image_id, bbox_list):
        gt_bboxes = self.dataset_coco.get_original_bboxes_list(image_id=image_id)
        sparse_targets = self.dataset_coco.get_original_category_sparse_list(image_id=image_id)

        # For each gt_bbox calculate ious with candidates
        bboxes_ious = []
        for bbox in gt_bboxes:
            ious = BoundingBoxTools.ious(bbox_list, bbox)
            ious_temp = np.zeros(shape=(len(ious)), dtype=np.float32)
            # Use 0.5 to use max
            ious_temp = np.where(np.asarray(ious) > self.threshold_iou_roi, 1, ious_temp)
            ious_temp[np.argmax(ious)] = 1
            bboxes_ious.append(ious_temp)

        # For each gt_box, determine the box reg target
        original_img = self.generate_train_input_one(image_id)
        input_images = []
        input_box_filtered_by_iou = []
        target_classes = []
        target_bbox_reg = []
        for index_gt, bbox_ious in enumerate(bboxes_ious):
            candidate_boxes = np.asarray(bbox_list)[np.where(bbox_ious == 1)]
            n = candidate_boxes.shape[0]
            for i in range(n):
                input_box_filtered_by_iou.append(candidate_boxes[i].astype(np.float32))
                box_reg = BoundingBoxTools.bbox_regression_target(pred_boxes=candidate_boxes[i].reshape((1, 4)),
                                                                  gt_box=gt_bboxes[index_gt])
                target_bbox_reg.append(box_reg.ravel())
                target_classes.append(sparse_targets[index_gt])
                input_images.append(original_img.astype(np.float32))

        for index_gt, bbox_gt in enumerate(gt_bboxes):
            input_images.append(original_img.astype(np.float32))
            input_box_filtered_by_iou.append(bbox_gt.astype(np.float32))
            target_classes.append(sparse_targets[index_gt])
            target_bbox_reg.append(np.array([0, 0, 0, 0], dtype=np.float32))

        return np.asarray(input_images).astype(np.float32), np.asarray(input_box_filtered_by_iou), np.asarray(
            target_classes), np.asarray(target_bbox_reg)

    def _validate_bbox(self, image_id, bboxes):
        img1 = self.dataset_coco.get_original_image(image_id=image_id)
        for bbox in bboxes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img1 = cv2.rectangle(img1, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 4)
        plt.imshow(img1)
        plt.show()

    def _validate_masks(self, image_id):
        img1 = self.dataset_coco.get_original_image(image_id=image_id)
        temp_img = np.zeros(shape=img1.shape, dtype=np.uint8)
        masks = self.dataset_coco.get_original_segmentations_mask_list(image_id=image_id)
        for mask in masks:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            temp_img[:, :, 0][mask.astype(bool)] = color[0]
            temp_img[:, :, 1][mask.astype(bool)] = color[1]
            temp_img[:, :, 2][mask.astype(bool)] = color[2]
        img1 = (img1 * 0.5 + temp_img * 0.5).astype(np.uint8)
        plt.imshow(img1, cmap='gray')
        plt.show()