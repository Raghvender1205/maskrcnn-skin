import numpy as np


class BoundingBoxTools:
    """
    format of bounding boxes is List for ious
    """

    @classmethod
    def ious(cls, boxes: list, box1_target):
        """
        Calculate IoUs
        """
        # box axis format: (x1, y1, x2, y2)
        box_b_area = (box1_target[2] - box1_target[0] + 1) * (box1_target[3] - box1_target[1] + 1)
        ious = []
        for box in boxes:
            # get (x, y) coordinates of the intersection rectangle
            x_a = max(box[0], box1_target[0])
            y_a = max(box[1], box1_target[1])
            x_b = min(box[2], box1_target[2])
            y_b = min(box[3], box1_target[3])

            # Area of intersection rectangle
            intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
            # Area of both the prediction and ground-truth rectangles
            box_a_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

            # IoU -> Intersection Over Union
            ious.append(intersection_area / float(box_a_area + box_b_area - intersection_area))

        return ious

    @classmethod
    def bbox_regression_target(cls, pred_boxes, gt_box):
        """
        :param pred_boxes: (Expected bbox) numpy array of shape (BS, x1, y1, x2, y2)
        :param gt_box: (Ground Truth) numpy array of shape (BS, x1, y1, x2, y2)
        :return: transforms
        """
        gt_boxes = np.zeros(shape=pred_boxes.shape) + gt_box
        reg_target = np.zeros(shape=pred_boxes.shape)
        ex_boxes_xywh = cls.xxyy2xywh(pred_boxes)
        gt_boxes_xywh = cls.xxyy2xywh(gt_boxes)

        # Makes sure the target label is [-1, 1], which can only be achieved when iou > 0.7
        reg_target[:, 0] = (gt_boxes_xywh[:, 0] - ex_boxes_xywh[:, 0]) / ex_boxes_xywh[:, 2]
        reg_target[:, 1] = (gt_boxes_xywh[:, 1] - ex_boxes_xywh[:, 1]) / ex_boxes_xywh[:, 3]
        reg_target[:, 2] = np.log(gt_boxes_xywh[:, 2] / ex_boxes_xywh[:, 2])
        reg_target[:, 3] = np.log(gt_boxes_xywh[:, 3] / ex_boxes_xywh[:, 3])

        return reg_target

    @classmethod
    def bbox_reg2truebox(cls, base_boxes, regs):
        """
        :param base_boxes: input shape => (N, 4)
        :param regs: input shape => (N, 4)
        """
        box_after_reg = np.zeros(shape=base_boxes.shape)
        base_box_xywh = cls.xxyy2xywh(base_boxes)

        box_after_reg[:, 0] = regs[:, 0] * base_box_xywh[:, 2] + base_box_xywh[:, 0]
        box_after_reg[:, 1] = regs[:, 1] * base_box_xywh[:, 3] + base_box_xywh[:, 1]
        box_after_reg[:, 2] = np.exp(regs[:, 2]) * base_box_xywh[:, 2]
        box_after_reg[:, 3] = np.exp(regs[:, 3]) * base_box_xywh[:, 3]
        box_after_reg = cls.xywh2xxyy(box_after_reg)

        return box_after_reg.astype(np.int32)  # int -> int32

    @classmethod
    def xxyy2xywh(cls, boxes):
        """
        Convert xxyy coordinates to xywh coordinates
        :param boxes:
        :return:
        """
        xywh = np.zeros(shape=boxes.shape)
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0] + 1
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1] + 1
        xywh[:, 0] = boxes[:, 0] + xywh[:, 2] / 2
        xywh[:, 1] = boxes[:, 1] + xywh[:, 3] / 2

        return xywh.astype(np.int32)

    @classmethod
    def xywh2xxyy(cls, boxes):
        """
        Convert xywh coordinates to xxyy coordinates
        :param boxes:
        :return:
        """
        xyxy = np.zeros(shape=boxes.shape)
        xyxy[:, 0] = boxes[:, 0] - (boxes[:, 2] - 1) / 2
        xyxy[:, 1] = boxes[:, 1] - (boxes[:, 3] - 1) / 2
        xyxy[:, 2] = boxes[:, 0] + (boxes[:, 2] - 1) / 2
        xyxy[:, 3] = boxes[:, 1] + (boxes[:, 3] - 1) / 2

        return xyxy.astype(np.int32)

    @classmethod
    def clip_boxes(cls, boxes, img_shape):
        x_max, y_max = img_shape[0], img_shape[1]
        boxes[:, 0][boxes[:, 0] < 0] = 0
        boxes[:, 1][boxes[:, 1] < 0] = 0
        boxes[:, 2][boxes[:, 2] < 0] = 1
        boxes[:, 3][boxes[:, 3] < 0] = 1
        boxes[:, 0][boxes[:, 0] > x_max] = x_max - 1
        boxes[:, 1][boxes[:, 1] > y_max] = y_max - 1
        boxes[:, 2][boxes[:, 2] > x_max] = x_max
        boxes[:, 3][boxes[:, 3] > y_max] = y_max

        return boxes

    @classmethod
    def clip_boxes_cls(cls, boxes, img_shape):
        boxes[:, 0::4].clamp_(0, img_shape[1] - 1)
        boxes[:, 1::4].clamp_(0, img_shape[0] - 1)
        boxes[:, 2::4].clamp_(0, img_shape[1] - 1)
        boxes[:, 3::4].clamp_(0, img_shape[0] - 1)

        return boxes
