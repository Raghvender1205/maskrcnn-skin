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
    inputs, anchor_targets, bbox_regression_targets = model.train_data_generator.generate_train_data_rpn_one(image_ids[0])
    logger.info(f"{inputs.shape}, {anchor_targets.shape}, {bbox_regression_targets.shape}")
    image = np.reshape(inputs[0, :, :, :], (1, SkinConfig.IMG_ORIGINAL_SHAPE[0], SkinConfig.IMG_ORIGINAL_SHAPE[1], 3))
    # Get proposed region boxes
    rpn_anchor_pred, rpn_bbox_regression_pred = model.rpn.process_image(image)
    proposed_boxes = model.rpn._propose_boxes(rpn_anchor_pred, rpn_bbox_regression_pred,
                                              model.anchor_candidates,
                                              model.anchor_candidate_generator.h, model.anchor_candidate_generator.w,
                                              model.anchor_candidate_generator.n_anchors,
                                              SkinConfig.ANCHOR_N_PROPOSAL, SkinConfig.ANCHOR_THRESHOLD)
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
    test_faster_rcnn_output()