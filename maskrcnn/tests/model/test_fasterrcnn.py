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
from maskrcnn.model import FasterRCNNModel, SkinConfig

model = FasterRCNNModel()

# def test_loss_function():
#     inputs, anchor_targets, bbox_targets = model.train_data_generator.generate_train_data_rpn_all()
#     logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_targets.shape}")
#     input1 = np.reshape(inputs[0, :, :, :], (1, 512, 512, 3))
#     anchor1 = np.reshape(anchor_targets[0, :, :, :], (1, 23, 40, 9))
#     anchor2 = tf.convert_to_tensor(anchor1)
#     anchor2 = tf.dtypes.cast(anchor2, tf.int32)
#     anchor2 = tf.one_hot(anchor2, 2, axis=-1)
#     logger.info(anchor1)
#     bbox1 = np.reshape(bbox_targets[0, :, :, :, :], (1, 23, 40, 9, 4))
#     loss = model.rpn._rpn_loss(anchor1, bbox1, anchor2, bbox1)
#     logger.info('loss: {}'.format(loss))


def test_loss_function():
    inputs, anchor_targets, bbox_targets = model.train_data_generator.generate_train_data_rpn_all()
    logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_targets.shape}")
    input1 = np.reshape(inputs[0, :, :, :], (1, 512, 512, 3))
    anchor1 = np.reshape(anchor_targets[0, :, :, :], (1, 16, 16, 9))  # Changed shape to match actual data
    anchor2 = tf.convert_to_tensor(anchor1)
    anchor2 = tf.dtypes.cast(anchor2, tf.int32)
    anchor2 = tf.one_hot(anchor2, 2, axis=-1)
    # Ensure this matches the anchor1 reshaping if needed
    bbox1 = np.reshape(bbox_targets[0, :, :, :, :], (1, 16, 16, 9, 4))
    loss = model.rpn._rpn_loss(anchor1, bbox1, anchor2, bbox1)
    logger.info('loss: {}'.format(loss))


def test_faster_rcnn_output():
    image_ids = model.train_data_generator.dataset_coco.image_ids
    inputs, anchor_targets, bbox_regression_targets = model.train_data_generator.generate_train_data_rpn_one(
        image_ids[0])
    logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_regression_targets.shape}")
    image = np.reshape(inputs[0, :, :, :], (1, SkinConfig.IMG_ORIGINAL_SHAPE[0], SkinConfig.IMG_ORIGINAL_SHAPE[1], 3))
    # Get proposed region boxes
    outputs = model.rpn.process_image(image)
    if isinstance(outputs, list):
        rpn_anchor_pred, rpn_bbox_regression_pred = outputs
    else:
        raise ValueError("Model outputs are not expected list format")
    if rpn_anchor_pred is None or rpn_bbox_regression_pred is None:
        logger.error("RPN processing failed to produce valid outputs.")
    print("Output structure:", type(rpn_anchor_pred), rpn_anchor_pred.shape)
    print("Output structure:", type(rpn_bbox_regression_pred), rpn_bbox_regression_pred.shape)
    proposed_boxes = model.rpn._proposal_boxes(rpn_anchor_pred, rpn_bbox_regression_pred,
                                               model.anchor_candidates,
                                               model.anchor_candidate_generator.h, model.anchor_candidate_generator.w,
                                               model.anchor_candidate_generator.n_anchors,
                                               SkinConfig.ANCHOR_N_PROPOSAL, SkinConfig.ANCHOR_THRESHOLD)

    if proposed_boxes is None:
        logger.error("proposed_boxes is None")
    else:
        logger.info(f"proposed_boxes: {proposed_boxes.shape}")

    # Processing boxes with RoI header
    pred_class, pred_box_regression = model.roi.process_image([image, proposed_boxes])
    pred_class_sparse = np.argmax(a=pred_class[:, :], axis=1)
    pred_class_sparse_value = np.max(a=pred_class[:, :], axis=1)
    logger.info(f"{pred_class}, {pred_box_regression}")
    logger.info(f"pred_class_sparse: {pred_class_sparse}")
    logger.info(f"{np.argmax(proposed_boxes)} {np.argmax(pred_box_regression)}")
    final_box = BoundingBoxTools.bbox_reg2truebox(base_boxes=proposed_boxes, regs=pred_box_regression)
    final_box = BoundingBoxTools.clip_boxes(final_box, img_shape=model.img_shape)

    # Output to official coco results json
    temp_output_to_file = []
    for i in range(pred_class_sparse.shape[0]):
        temp_category = model.train_data_generator.dataset_coco.get_category_from_sparse(pred_class_sparse[i])
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
    model.cocotool.draw_bboxes(original_image=image[0], bboxes=final_box.tolist(), show=True, save_file=True,
                               path=SkinConfig.PATH_DEBUG_IMG, save_name="PredRoIBoxes")

    # NMS
    final_box_tmp = np.array(final_box).astype(np.int32)
    nms_boxes_list = []
    while final_box_tmp.shape[0] > 0:
        ious = model.nms_loop_np(final_box_tmp)
        nms_boxes_list.append(
            final_box_tmp[0, :])
        final_box_tmp = final_box_tmp[ious < SkinConfig.RPN_NMS_THRESHOLD]
    logger.debug(f'Number of boxes after NMS: {len(nms_boxes_list)}')
    model.cocotool.draw_bboxes(original_image=image[0], bboxes=nms_boxes_list, show=True, save_file=True,
                               path=SkinConfig.PATH_DEBUG_IMG, save_name="PredRoIBoxesNMS")


def test_proposal_visualization():
    image_ids = model.train_data_generator.dataset_coco.image_ids
    inputs, anchor_targets, bbox_regression_targets = model.train_data_generator.generate_train_data_rpn_one(image_ids[0])
    logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_regression_targets.shape}")
    input1 = np.reshape(inputs[0, :, :, :], (1, SkinConfig.IMG_ORIGINAL_SHAPE[0], SkinConfig.IMG_ORIGINAL_SHAPE[1], 3))
    rpn_anchor_pred, rpn_bbox_regression_pred = model.rpn.process_image(input1)
    logger.info(f"rpn_anchor_pred: {rpn_anchor_pred.shape}, rpn_bbox_regression_pred: {rpn_bbox_regression_pred.shape}")

    # Selection Part
    rpn_anchor_pred = tf.slice(rpn_anchor_pred, [0, 0, 0, 0, 1],
                               [1, model.anchor_candidate_generator.h, model.anchor_candidate_generator.w,
                                model.anchor_candidate_generator.n_anchors, 1])
    logger.info(f"rpn_anchor_pred: {rpn_anchor_pred.shape}, rpn_bbox_regression_pred: {rpn_bbox_regression_pred.shape}")
    # squeeze the pred of anchor and bbox_reg
    rpn_anchor_pred = tf.squeeze(rpn_anchor_pred)
    rpn_bbox_regression_pred = tf.squeeze(rpn_bbox_regression_pred)
    shape1 = tf.shape(rpn_anchor_pred)
    logger.info(f"rpn_anchor_pred_shape: {rpn_anchor_pred.shape}, rpn_bbox_regression_pred_shape: {rpn_bbox_regression_pred.shape}")
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
    base_boxes = tf.gather_nd(model.anchor_candidates, anchor_pred_top_indices)
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
        ious = model.nms_loop_np(final_box_tmp)
        nms_boxes_list.append(
            final_box_tmp[0, :]
        )
        final_box_tmp = final_box_tmp[ious < SkinConfig.RPN_NMS_THRESHOLD]
    logger.debug(f"Number of boxes after NMS: {len(nms_boxes_list)}")

    # Visualization
    # Clip the boxes
    logger.debug(f"Max value of final box: {np.max(final_box)}")
    final_box = BoundingBoxTools.clip_boxes(final_box, model.img_shape)

    original_boxes = model.cocotool.get_original_bboxes_list(image_id=model.cocotool.image_ids[0])
    model.cocotool.draw_bboxes(original_image=input1[0], bboxes=original_boxes, show=True, save_file=True,
                               path=SkinConfig.PATH_DEBUG_IMG, save_name="1GTBoxes")
    target_anchor_boxes, target_classes = model.train_data_generator.generate_target_anchor_bboxes_classes_for_debug(
        image_id=model.cocotool.image_ids[0])
    model.cocotool.draw_bboxes(original_image=input1[0], bboxes=target_anchor_boxes, show=True, save_file=True,
                               path=SkinConfig.PATH_DEBUG_IMG, save_name="2TrueAnchorBoxes")
    model.cocotool.draw_bboxes(original_image=input1[0], bboxes=base_boxes.tolist(), show=True, save_file=True,
                               path=SkinConfig.PATH_DEBUG_IMG, save_name="3PredAnchorBoxes")
    model.cocotool.draw_bboxes(original_image=input1[0], bboxes=final_box.tolist(), show=True, save_file=True,
                               path=SkinConfig.PATH_DEBUG_IMG, save_name="4PredRegressionBoxes")
    model.cocotool.draw_bboxes(original_image=input1[0], bboxes=nms_boxes_list, show=True, save_file=True,
                               path=SkinConfig.PATH_DEBUG_IMG, save_name="5PredNMSBoxes")


def test_total_visualization():
    input_images, target_anchor_bboxes, target_classes = model.train_data_generator.generate_train_data_roi_one(
        model.train_data_generator.dataset_coco.image_ids[0],
        model.train_data_generator.generate_candidate_anchors.anchor_candidates_list)

    input_images, target_anchor_bboxes, target_classes = (np.asarray(input_images).astype(np.float32),
                                                          np.asarray(target_anchor_bboxes), np.asarray(target_classes))

    input_images2 = input_images[:1].astype(np.float32)
    logger.info(input_images2.shape)
    target_anchor_bboxes2 = target_anchor_bboxes[:1].astype(np.float32)
    logger.info(target_anchor_bboxes2.shape)
    class_header, box_regressor_head = model.roi.process_image([input_images2, target_anchor_bboxes2])
    logger.info(class_header.shape, box_regressor_head.shape)
    logger.info(class_header)


if __name__ == '__main__':
    # test_loss_function()
    # test_total_visualization()
    # test_faster_rcnn_output()
    test_proposal_visualization()