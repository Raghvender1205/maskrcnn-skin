import numpy as np
import tensorflow as tf

from maskrcnn.utils.bbox_tools import BoundingBoxTools


class BoundingBoxToolsTF:
    @classmethod
    def _ious(cls, boxes_np, bbox_1_target):
        """
        Box axis format is (x1, y1, x2, y2)
        :param boxes_np: (?, 4)
        :param bbox_1_target:
        :return:
        """
        shape = boxes_np.shape
        box_b_area = (bbox_1_target[2] - bbox_1_target[0] + 1) * (bbox_1_target[3] - bbox_1_target[1] + 1)
        ious = np.zeros(shape=shape[0])
        for i in range(shape[0]):
            # Get (x, y) coordinates of the intersection rectangle
            box = boxes_np[i]
            x_a = max(box[0], bbox_1_target[0])
            y_a = max(box[1], bbox_1_target[1])
            x_b = min(box[2], bbox_1_target[2])
            y_b = min(box[3], bbox_1_target[3])

            # Area of intersection rectangle
            intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

            # Area of both prediction and ground truth rectangles
            box_a_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

            # IoU
            ious[i] = (intersection_area / float(box_a_area + box_b_area - intersection_area))

        return ious

    @classmethod
    def ious(cls, boxes_np, bbox_1_target):
        ious = tf.numpy_function(cls._ious, [boxes_np, bbox_1_target], tf.float32)

        return ious

    @classmethod
    def bbox_regression_target(cls, pred_boxes, gt_box):
        reg_target = tf.numpy_function(BoundingBoxTools.bbox_regression_target, [pred_boxes, gt_box], tf.float32)

        return reg_target

    @classmethod
    def bbox_reg2truebox(cls, base_boxes, regs):
        truebox = tf.numpy_function(BoundingBoxTools.bbox_reg2truebox, [base_boxes, regs], tf.float32)

        return truebox

    @classmethod
    def xxyy2xywh(cls, boxes):
        xywh = tf.numpy_function(BoundingBoxTools.xxyy2xywh, [boxes], tf.float32)

        return xywh

    @classmethod
    def xywh2xxyy(cls, boxes):
        xyxy = tf.numpy_function(BoundingBoxTools.xywh2xxyy, [boxes], tf.int32)

        return xyxy

    @classmethod
    def clip_boxes(cls, boxes, img_shape):
        boxes2 = tf.numpy_function(BoundingBoxTools.clip_boxes, [boxes, img_shape], tf.int32)

        return boxes2