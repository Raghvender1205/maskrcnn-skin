from maskrcnn.utils.cocotools import COCOTools
from maskrcnn.utils import GenerateCandidateAnchors, BoundingBoxTools, DataGenerator
from maskrcnn.configs.fasterrcnn_config import FasterRCNNConfig

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def test():
    base_path = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data"
    image_folder_path = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/images"
    image_id = '6'  # For testing.
    data1 = COCOTools(json_file=f"{base_path}/train/annotations.json",
                      image_folder_path=image_folder_path, resized_shape=FasterRCNNConfig.IMG_RESIZED_SHAPE)
    img1 = data1.get_original_image(image_id=image_id)
    # print(data1.images)
    bboxes = data1.get_original_bboxes_list(image_id=image_id)
    print(bboxes)
    for bbox in bboxes:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img1, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 4)
    plt.imshow(img1)
    plt.show()

    g1 = GenerateCandidateAnchors(ratios=FasterRCNNConfig.ANCHOR_RATIOS)
    print(len(g1.anchor_candidates_list))
    ious = BoundingBoxTools.ious(g1.anchor_candidates_list, bboxes[0])
    ious[np.argmax(ious)] = 1
    print(len(ious))
    ious_np = np.reshape(ious, newshape=(23, 40, 9))
    index = np.where(ious_np == 1)
    print(index)


def test2():
    base_path = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data"
    image_folder_path = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/images"
    image_id = '6'  # For testing.
    t1 = DataGenerator(file=f"{base_path}/train/annotations.json",
                       image_folder_path=image_folder_path,
                       anchor_base_size=FasterRCNNConfig.ANCHOR_BASE_SIZE,
                       ratios=FasterRCNNConfig.ANCHOR_RATIOS,
                       scales=FasterRCNNConfig.ANCHOR_SCALES,
                       n_anchors=FasterRCNNConfig.N_ANCHORS)
    bboxes = t1.dataset_coco.get_original_bboxes_list(image_id=image_id)
    t1._validate_bbox(image_id=image_id, bboxes=bboxes)
    t1._validate_masks(image_id=image_id)
    t1.generate_train_target_box_regression_for_rpn(image_id=image_id)


if __name__ == "__main__":
    # test()
    test2()
