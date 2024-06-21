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
    DATA_JSON_FILE = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/annotations.json"
    PATH_IMAGES = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/images"
    PATH_DEBUG_IMG = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/images/29900.png"


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

    def test_loss_function(self):
        inputs, anchor_targets, bbox_targets = self.train_data_generator.generate_train_data_rpn_all()
        logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_targets.shape}")
        input1 = np.reshape(inputs[0, :, :, :], (1, 512, 512, 3))
        anchor1 = np.reshape(anchor_targets[0, :, :, :], (1, 16, 16, 9))  # Changed shape to match actual data
        anchor2 = tf.convert_to_tensor(anchor1)
        anchor2 = tf.dtypes.cast(anchor2, tf.int32)
        anchor2 = tf.one_hot(anchor2, 2, axis=-1)
        # Ensure this matches the anchor1 reshaping if needed
        bbox1 = np.reshape(bbox_targets[0, :, :, :, :], (1, 16, 16, 9, 4))
        loss = self.rpn._rpn_loss(anchor1, bbox1, anchor2, bbox1)
        logger.info('loss: {}'.format(loss))

    def test_faster_rcnn_output(self):
        image_ids = self.train_data_generator.dataset_coco.image_ids
        inputs, anchor_targets, bbox_regression_targets = self.train_data_generator.generate_train_data_rpn_one(
            image_ids[0])
        logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_regression_targets.shape}")
        image = np.reshape(inputs[0, :, :, :],
                           (1, SkinConfig.IMG_ORIGINAL_SHAPE[0], SkinConfig.IMG_ORIGINAL_SHAPE[1], 3))
        # Get proposed region boxes
        outputs = self.rpn.process_image(image)
        if isinstance(outputs, list):
            rpn_anchor_pred, rpn_bbox_regression_pred = outputs
        else:
            raise ValueError("Model outputs are not expected list format")
        if rpn_anchor_pred is None or rpn_bbox_regression_pred is None:
            logger.error("RPN processing failed to produce valid outputs.")
        print("Output structure:", type(rpn_anchor_pred), rpn_anchor_pred.shape)
        print("Output structure:", type(rpn_bbox_regression_pred), rpn_bbox_regression_pred.shape)
        proposed_boxes = self.rpn._proposal_boxes(rpn_anchor_pred, rpn_bbox_regression_pred,
                                                   self.anchor_candidates,
                                                   self.anchor_candidate_generator.h,
                                                   self.anchor_candidate_generator.w,
                                                   self.anchor_candidate_generator.n_anchors,
                                                   SkinConfig.ANCHOR_N_PROPOSAL, SkinConfig.ANCHOR_THRESHOLD)

        if proposed_boxes is None:
            logger.error("proposed_boxes is None")
        else:
            logger.info(f"proposed_boxes: {proposed_boxes.shape}")

        # Processing boxes with RoI header
        pred_class, pred_box_regression = self.roi.process_image([image, proposed_boxes])
        pred_class_sparse = np.argmax(a=pred_class[:, :], axis=1)
        pred_class_sparse_value = np.max(a=pred_class[:, :], axis=1)
        logger.info(f"{pred_class}, {pred_box_regression}")
        logger.info(f"pred_class_sparse: {pred_class_sparse}")
        logger.info(f"{np.argmax(proposed_boxes)} {np.argmax(pred_box_regression)}")
        final_box = BoundingBoxTools.bbox_reg2truebox(base_boxes=proposed_boxes, regs=pred_box_regression)
        final_box = BoundingBoxTools.clip_boxes(final_box, img_shape=self.img_shape)

        # Output to official coco results json
        temp_output_to_file = []
        for i in range(pred_class_sparse.shape[0]):
            temp_category = self.train_data_generator.dataset_coco.get_category_from_sparse(pred_class_sparse[i])
            temp_output_to_file.append({
                'image_id': f"{image_ids[0]}",
                "bbox": [final_box[i][0].item(), final_box[i][1].item(), final_box[i][2].item(),
                         final_box[i][3].item()],
                "score": pred_class_sparse_value[i].item(),
                "category": f"{temp_category}"
            })

        with open("results.pkl.bbox.json", "w") as f:
            json.dump(temp_output_to_file, f, indent=4)
        logger.info(f"{final_box[pred_class_sparse_value > 0.9]}")
        final_box = final_box[pred_class_sparse_value > 0.9]
        self.cocotool.draw_bboxes(original_image=image[0], bboxes=final_box.tolist(), show=True, save_file=True,
                                   path=SkinConfig.PATH_DEBUG_IMG, save_name="PredRoIBoxes")

        # NMS
        final_box_tmp = np.array(final_box).astype(np.int32)
        nms_boxes_list = []
        while final_box_tmp.shape[0] > 0:
            ious = self.nms_loop_np(final_box_tmp)
            nms_boxes_list.append(
                final_box_tmp[0, :])
            final_box_tmp = final_box_tmp[ious < SkinConfig.RPN_NMS_THRESHOLD]
        logger.debug(f'Number of boxes after NMS: {len(nms_boxes_list)}')
        self.cocotool.draw_bboxes(original_image=image[0], bboxes=nms_boxes_list, show=True, save_file=True,
                                   path=SkinConfig.PATH_DEBUG_IMG, save_name="PredRoIBoxesNMS")

    def test_proposal_visualization(self):
        image_ids = self.train_data_generator.dataset_coco.image_ids
        inputs, anchor_targets, bbox_regression_targets = self.train_data_generator.generate_train_data_rpn_one(
            image_ids[0])
        logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_regression_targets.shape}")
        input1 = np.reshape(inputs[0, :, :, :],
                            (1, SkinConfig.IMG_ORIGINAL_SHAPE[0], SkinConfig.IMG_ORIGINAL_SHAPE[1], 3))
        rpn_anchor_pred, rpn_bbox_regression_pred = self.rpn.process_image(input1)
        logger.info(
            f"rpn_anchor_pred: {rpn_anchor_pred.shape}, rpn_bbox_regression_pred: {rpn_bbox_regression_pred.shape}")

        # Selection Part
        rpn_anchor_pred = tf.slice(rpn_anchor_pred, [0, 0, 0, 0, 1],
                                   [1, self.anchor_candidate_generator.h, self.anchor_candidate_generator.w,
                                    self.anchor_candidate_generator.n_anchors, 1])
        logger.info(
            f"rpn_anchor_pred: {rpn_anchor_pred.shape}, rpn_bbox_regression_pred: {rpn_bbox_regression_pred.shape}")
        # squeeze the pred of anchor and bbox_reg
        rpn_anchor_pred = tf.squeeze(rpn_anchor_pred)
        rpn_bbox_regression_pred = tf.squeeze(rpn_bbox_regression_pred)
        shape1 = tf.shape(rpn_anchor_pred)
        logger.info(
            f"rpn_anchor_pred_shape: {rpn_anchor_pred.shape}, rpn_bbox_regression_pred_shape: {rpn_bbox_regression_pred.shape}")
        # flatten the anchor pred to get top N values and indices
        rpn_anchor_pred = tf.reshape(rpn_anchor_pred, (-1,))
        n_anchor_proposal = SkinConfig.ANCHOR_N_PROPOSAL

        top_values, top_indices = tf.math.top_k(rpn_anchor_pred, n_anchor_proposal)
        top_indices = tf.gather_nd(top_indices, tf.where(tf.greater(top_values, SkinConfig.ANCHOR_THRESHOLD)))
        top_values = tf.gather_nd(top_values, tf.where(tf.greater(top_values, SkinConfig.ANCHOR_THRESHOLD)))
        logger.debug(f"Top Values: {top_values}")

        top_indices = tf.reshape(top_indices, (-1, 1))
        logger.debug(f"Top Indices: {top_indices}")
        update_value = tf.math.add(top_values, 1)
        rpn_anchor_pred = tf.tensor_scatter_nd_update(rpn_anchor_pred, top_indices, update_value)
        rpn_anchor_pred = tf.reshape(rpn_anchor_pred, shape1)

        # Find base boxes
        anchor_pred_top_indices = tf.where(tf.greater(rpn_anchor_pred, 1))
        logger.debug(f"original_indices shape: {anchor_pred_top_indices.shape}")
        logger.debug(f"original_indices: {anchor_pred_top_indices}")
        base_boxes = tf.gather_nd(self.anchor_candidates, anchor_pred_top_indices)
        logger.debug(f"base_boxes shape: {base_boxes.shape}")
        logger.debug(f"base_boxes: {base_boxes}")
        base_boxes = np.array(base_boxes)

        # find bbox regressions
        # flatten the bbox_regression by last dim to use top_indices to get final_box_regression
        rpn_bbox_regression_pred_shape = tf.shape(rpn_bbox_regression_pred)
        rpn_bbox_regression_pred = tf.reshape(rpn_bbox_regression_pred, (-1, rpn_bbox_regression_pred_shape[-1]))
        logger.debug(f"rpn_bbox_regression_pred shape: {rpn_bbox_regression_pred.shape}")
        final_box_regression = tf.gather_nd(rpn_bbox_regression_pred, top_indices)
        logger.debug(f"Final box regression values: {final_box_regression}")

        # Convert to np for plot
        final_box_regression = np.array(final_box_regression)
        logger.debug(f"Final box regression shape: {final_box_regression.shape}")
        logger.debug(f"Max value of final box regression: {np.max(final_box_regression)}")
        final_box = BoundingBoxTools.bbox_reg2truebox(base_boxes=base_boxes, regs=final_box_regression)

        # NMS
        final_box_tmp = np.array(final_box).astype(np.int32)
        nms_boxes_list = []
        while final_box_tmp.shape[0] > 0:
            ious = self.nms_loop_np(final_box_tmp)
            nms_boxes_list.append(
                final_box_tmp[0, :]
            )
            final_box_tmp = final_box_tmp[ious < SkinConfig.RPN_NMS_THRESHOLD]
        logger.debug(f"Number of boxes after NMS: {len(nms_boxes_list)}")

        # Visualization
        # Clip the boxes
        logger.debug(f"Max value of final box: {np.max(final_box)}")
        final_box = BoundingBoxTools.clip_boxes(final_box, self.img_shape)

        original_boxes = self.cocotool.get_original_bboxes_list(image_id=self.cocotool.image_ids[0])
        self.cocotool.draw_bboxes(original_image=input1[0], bboxes=original_boxes, show=True, save_file=True,
                                   path=SkinConfig.PATH_DEBUG_IMG, save_name="1GTBoxes")
        target_anchor_boxes, target_classes = self.train_data_generator.generate_target_anchor_bboxes_classes_for_debug(
            image_id=self.cocotool.image_ids[0])
        self.cocotool.draw_bboxes(original_image=input1[0], bboxes=target_anchor_boxes, show=True, save_file=True,
                                   path=SkinConfig.PATH_DEBUG_IMG, save_name="2TrueAnchorBoxes")
        self.cocotool.draw_bboxes(original_image=input1[0], bboxes=base_boxes.tolist(), show=True, save_file=True,
                                   path=SkinConfig.PATH_DEBUG_IMG, save_name="3PredAnchorBoxes")
        self.cocotool.draw_bboxes(original_image=input1[0], bboxes=final_box.tolist(), show=True, save_file=True,
                                   path=SkinConfig.PATH_DEBUG_IMG, save_name="4PredRegressionBoxes")
        self.cocotool.draw_bboxes(original_image=input1[0], bboxes=nms_boxes_list, show=True, save_file=True,
                                   path=SkinConfig.PATH_DEBUG_IMG, save_name="5PredNMSBoxes")

    def test_total_visualization(self):
        try:
            image_id = self.train_data_generator.dataset_coco.image_ids[0]
            bbox_list = self.train_data_generator.generate_candidate_anchors.anchor_candidates_list

            input_images, input_box_filtered_by_iou, target_classes, target_bbox_reg = self.train_data_generator.generate_train_data_roi_one(
                image_id, bbox_list)

            # Log the shapes for debugging
            logger.info(f"Input Images Shape: {input_images.shape}")
            logger.info(f"Input Boxes Shape: {input_box_filtered_by_iou.shape}")
            logger.info(f"Target Classes Shape: {target_classes.shape}")
            logger.info(f"Target BBox Regression Shape: {target_bbox_reg.shape}")

            # Proceed with further processing or visualization
            # Example: Process with RoI network
            class_header, box_regressor_head = self.roi.process_image([input_images, input_box_filtered_by_iou])
            logger.info(f"Class Header Shape: {class_header.shape}, Box Regressor Shape: {box_regressor_head.shape}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def train_rpn_roi(self):
        # TODO: use the output of RPN to train ROI
        image_ids = self.train_data_generator.dataset_coco.image_ids
        for epoch in range(SkinConfig.EPOCH):
            logger.info(f"Epoch {epoch}")
            temp_image_ids = random.choices(population=image_ids, weights=None, k=8)

            for image_id in image_ids:
                # Train ROI
                input_img, input_box_filtered_by_iou, target_classes, target_bbox_regression = self.train_data_generator.generate_train_data_roi_one(
                    image_id, self.train_data_generator.generate_candidate_anchors.anchor_candidates_list
                )
                n_box = input_img.shape[0]
                # backbone self only be trained once to balance RPN and ROI training
                for j in range(n_box):
                    self.roi.train_step_backbone(input_img[j:j + 1] + 1, input_box_filtered_by_iou[j:j + 1],
                                                  target_classes[j:j + 1], target_bbox_regression[j:j + 1])

                # Train RPN backbone
                inputs, anchor_targets, bbox_regression_targets = self.train_data_generator.generate_train_data_rpn_one(
                    image_id)
                self.rpn.train_step_backbone(inputs, anchor_targets, bbox_regression_targets)

                # Train RPN Header
                for j in range(n_box):
                    self.roi.train_step_backbone(input_img[0:1], input_box_filtered_by_iou[j:j + 1],
                                                  target_classes[j:j + 1], target_bbox_regression[j:j + 1])

                # Train RPN with RPN proposed boxes
                if epoch > 10:
                    rpn_anchor_pred, rpn_bbox_regression_pred = self.rpn.process_image(input_img[:1])
                    proposed_boxes = self.rpn._proposal_boxes(rpn_anchor_pred, rpn_bbox_regression_pred,
                                                               self.anchor_candidates,
                                                               self.anchor_candidate_generator.h,
                                                               self.anchor_candidate_generator.w,
                                                               self.anchor_candidate_generator.n_anchors,
                                                               SkinConfig.ANCHOR_N_PROPOSAL,
                                                               SkinConfig.ANCHOR_THRESHOLD)
                    if len(list(proposed_boxes.tolist())) == 0:
                        continue
                    input_img, input_box_filtered_by_iou, target_classes, target_bbox_regression = self.train_data_generator.generate_train_data_roi_one(
                        image_id, proposed_boxes.tolist()
                    )
                    for j in range(n_box):
                        self.roi.train_step_header(input_img[j:j + 1], input_box_filtered_by_iou[j:j + 1],
                                                    target_classes[j:j + 1], target_bbox_regression[j:j + 1])

            # Train RPN header first
            for image_id in image_ids:
                inputs, anchor_targets, bbox_regression_targets = self.train_data_generator.generate_train_data_rpn_one(
                    image_id
                )
                self.rpn.train_step_header(inputs, anchor_targets, bbox_regression_targets)

    def save_weight(self):
        self.rpn.save_model(SkinConfig.PATH_MODEL)
        self.roi.save_header(SkinConfig.PATH_MODEL)
        self.backbone.save_weight(SkinConfig.PATH_MODEL)

    def load_weight(self):
        self.rpn.load_model(SkinConfig.PATH_MODEL)
        self.roi.save_header(SkinConfig.PATH_MODEL)
        self.backbone.load_weight(SkinConfig.PATH_MODEL)