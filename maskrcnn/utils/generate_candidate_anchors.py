import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from maskrcnn.utils.generate_base_anchors import GenerateBaseAnchors


class GenerateCandidateAnchors:
    def __init__(self,
                 ratios: list,
                 base_size: int = 16,
                 scales=2 ** np.arange(3, 6),
                 img_shape: tuple = (512, 512, 3),
                 n_stage: int = 5,
                 n_anchors: int = 9):
        if not ratios:
            ratios = [0.5, 1, 2]
        self.img_shape = (img_shape[0], img_shape[1])
        self.n_stage_revert_factor = 2 ** n_stage
        self.base_anchors = GenerateBaseAnchors.generate_base_anchors(base_size=base_size, ratios=ratios, scales=scales)

        # Round up the number since tf.conv2d round strategy
        self.h = int(img_shape[0] / self.n_stage_revert_factor) + int((img_shape[0] % self.n_stage_revert_factor) > 0)
        self.w = int(img_shape[1] / self.n_stage_revert_factor) + int((img_shape[1] % self.n_stage_revert_factor) > 0)
        self.n_anchors = n_anchors
        self.anchor_candidates = self.gen_all_candidate_anchors(self.h, self.w, self.n_anchors, self.img_shape)
        self.anchor_candidates_list = list(np.reshape(self.anchor_candidates, newshape=(-1, 4)).tolist())

    def gen_all_candidate_anchors(self, h: int, w: int, n_anchors: int, img_shape: tuple):
        anchors = np.zeros(shape=(h, w, n_anchors, 4), dtype=np.int32)
        # Anchors axis format => (x1, y1, x2, y2)
        x_max = img_shape[0] - 1
        y_max = img_shape[1] - 1
        for x in range(h):
            for y in range(w):
                temp = self.base_anchors + np.array([x * self.n_stage_revert_factor - self.n_stage_revert_factor / 2,
                                                     y * self.n_stage_revert_factor - self.n_stage_revert_factor / 2,
                                                     x * self.n_stage_revert_factor - self.n_stage_revert_factor / 2,
                                                     y * self.n_stage_revert_factor - self.n_stage_revert_factor / 2])
                temp[:, 0][temp[:, 0] < 0] = 0
                temp[:, 1][temp[:, 1] < 0] = 0
                temp[:, 2][temp[:, 2] > x_max] = x_max
                temp[:, 3][temp[:, 3] > y_max] = y_max
                anchors[x, y, :, :] = temp

        return anchors

    def _validate_bbox(self, bboxes):
        img1 = np.zeros(shape=(self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8)
        for bbox in bboxes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img=img1, rec=(bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]), color=color,
                          thickness=4)
        plt.imshow(img1)
        plt.show()


def get_feature_map_h_w_with_n_stages(img_shape: tuple, n_stage: int):
    n_stage_revert_factor = 2 ** n_stage
    h = int(img_shape[0] / n_stage_revert_factor) + int((img_shape[0] % n_stage_revert_factor) > 0)
    w = int(img_shape[1] / n_stage_revert_factor) + int((img_shape[1] % n_stage_revert_factor) > 0)

    return h, w
