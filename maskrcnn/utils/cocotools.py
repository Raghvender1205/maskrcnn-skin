import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from loguru import logger
from maskrcnn.helpers import pycococreatortools


class COCOTools:
    def __init__(self, json_file: str, image_folder_path: str, resized_shape: tuple):
        self.segment_info = None
        self.load_coco_annotations(json_file, image_folder_path)
        self.RESIZE_FLAG = False
        self.resized_shape = resized_shape
        if not all(resized_shape):
            self.RESIZE_FLAG = True

    def load_coco_annotations(self, file: str, image_folder_path: str):
        self.image_folder_path = image_folder_path
        self.file = file
        with open(file, 'r') as f:
            dict1 = json.load(f)
        self.info = dict1.get('info', {})
        self.licenses = dict1.get("licenses", [])
        self.images = dict1["images"]
        self.annotations = dict1["annotations"]
        self.categories = dict1["categories"]
        self.image_ids = []
        for image in self.images:
            if image['id'] not in self.image_ids:
                self.image_ids.append(image['id'])

        cnt = 0
        self.category2sparsecategory_onehot = {}
        self.sparsecategory_onehot2category = []
        for category in self.categories:
            if category['id'] not in self.category2sparsecategory_onehot:
                self.category2sparsecategory_onehot[category['id']] = cnt
                self.sparsecategory_onehot2category.append(category['id'])
                cnt += 1

    def _resize_annotations(self):
        for image_id in self.image_ids:
            original_shape = self.get_image_shape(image_id)

    def draw_segm_from_coco_annotations(self, image_id, original_image,
                                        annotations: list, show: bool = False,
                                        save_file: bool = False):
        height, width = self.get_image_shape(image_id)
        if self.RESIZE_FLAG:
            height, width, _ = self.resized_shape
        bboxes_int = self.get_original_bboxes_list(image_id)
        tempimg = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        masks_pred = class_ids = self.get_segm_mask_from_coco_annotations(annotations, image_id)
        _, _, n_masks = masks_pred.shape
        for i in range(n_masks):
            color_random = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_random2 = color_random.tolist()
            cv2.rectangle(tempimg,
                          (bboxes_int[i][1], bboxes_int[i][0]),
                          (bboxes_int[i][3], bboxes_int[i][2]),
                          color_random2[0], 2)
            cv2.putText(tempimg, f'{i + 1}th Object_{class_ids[i]}',
                        (bboxes_int[i][3] + 10, bboxes_int[i][2]),
                        0, 0.3, color_random2[0])
            tempimg[:, :, 0][masks_pred[:, :, i]] = color_random[0, 0]
            tempimg[:, :, 1][masks_pred[:, :, i]] = color_random[0, 1]
            tempimg[:, :, 2][masks_pred[:, :, i]] = color_random[0, 2]
        original_image = (original_image * 0.5 + tempimg * 0.5).astype(np.uint8)
        plt.imshow(original_image)
        if show:
            plt.show()
        if save_file:
            plt.savefig(f'{os.getcwd()}/output_images/{image_id}.jpg', dpi=300)

    def draw_bboxes(self, original_image, bboxes,
                    show: bool = False, save_file: bool = False,
                    path: str = "", save_name: str = ""):
        height, width = original_image.shape[0], original_image.shape[1]
        tempimg = np.zeros((height, width, 3), dtype=np.uint8)
        for bbox in bboxes:
            color_random = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_random2 = color_random.tolist()
            cv2.rectangle(tempimg,
                          (bbox[1], bbox[0]),
                          (bbox[3], bbox[2]),
                          color_random2[0],
                          2)
        original_image = (original_image * 0.5 + tempimg * 0.5).astype(np.uint8)
        plt.imshow(original_image)
        if show:
            plt.show()
        if save_file:
            img_opencv = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename=f"{path}/{save_name}.jpg", img=img_opencv)

    def get_segm_mask_from_coco_annotations(self, annotations, image_id):
        segmentations = []
        height, width = self.get_image_shape(image_id)
        if self.RESIZE_FLAG:
            height, width, _ = self.resized_shape

        class_ids = []
        for annotation in annotations:
            if annotation['image_id'] == image_id and isinstance(annotation['segmentation'], list):
                segm_tmp = np.reshape(annotation['segmentation'], (-1, 2))
                if self.RESIZE_FLAG:
                    original_shape = self.get_image_shape(image_id)
                    # opencv format is (y, x), different from numpy (x, y)
                    segm_tmp = segm_tmp / np.asarray([original_shape[1], original_shape[0]]) * np.asarray(
                        [self.resized_shape[1], self.resized_shape[0]]
                    )
                segmentations.append(segm_tmp.astype(int))
                class_ids.append(annotation['category_id'])
        n_segmentations = len(segmentations)
        mask_temp = np.zeros((height, width, n_segmentations), dtype=np.uint8)
        for i in range(n_segmentations):
            temp_one_mask = np.zeros((height, width), dtype=np.uint8)
            contour = segmentations[i]
            cv2.fillPoly(img=temp_one_mask, pts=[contour], color=1)
            mask_temp[:, :, i] = temp_one_mask

        return mask_temp.astype(np.bool_), class_ids

    def get_original_bboxes_list(self, image_id):
        """
        OpenCV bbox format: (y, x, dy, dx)
        Output bbox format: (x1, y1, x2, y2)
        """
        bboxes = []
        for annotation in self.annotations:
            if str(annotation['image_id']) == str(image_id):
                bbox = np.array(annotation['bbox'], dtype=np.int32)
                bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]
                if self.RESIZE_FLAG:
                    original_shape = self.get_image_shape(image_id)
                    bbox[0] = bbox[0] / original_shape[0] * self.resized_shape[0]
                    bbox[1] = bbox[1] / original_shape[1] * self.resized_shape[1]
                    bbox[2] = bbox[2] / original_shape[0] * self.resized_shape[0]
                    bbox[3] = bbox[3] / original_shape[1] * self.resized_shape[1]
                bboxes.append(bbox)

        return bboxes

    def get_original_category_sparse_list(self, image_id):
        categories_sparse = []
        for annotation in self.annotations:
            if annotation['image_id'] == image_id:
                sparse = self.category2sparsecategory_onehot[annotation['category_id']]
                categories_sparse.append(sparse)

        return categories_sparse

    def get_category_from_sparse(self, num: int):
        if num >= len(self.category2sparsecategory_onehot):
            logger.error(f"Category index {num} is out of range. Total Categories: {len(self.sparsecategory_onehot2category)}")
            return "Unknown Category"

        return self.sparsecategory_onehot2category[num]

    def get_original_segmentations_mask_list(self, image_id):
        height, width = self.get_image_shape(image_id)
        masks = []
        for annotation in self.annotations:
            if annotation['image_id'] == image_id:
                img_temp = np.zeros(shape=(height, width), dtype=np.uint8)
                contour = np.reshape(annotation['segmentation'], newshape=(-1, 2)).astype(int)
                img_temp = cv2.fillPoly(img_temp, [contour], 1)
                masks.append(img_temp)

        return masks

    # def get_original_image(self, image_id):
    #     image_name = self.get_image_name(image_id)
    #     img = cv2.imread(f"{self.image_folder_path}/{image_name}")
    #     if self.RESIZE_FLAG:
    #         img = cv2.resize(img, (self.resized_shape[1], self.resized_shape[0]), interpolation=cv2.INTER_AREA)
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     return np.array(img_rgb)

    def get_original_image(self, image_id):
        image_name = self.get_image_name(image_id)
        full_path = os.path.join(self.image_folder_path, image_name)
        full_path = os.path.normpath(full_path)  # Normalize the path
        # print("Attempting to load image from:", full_path)  e
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
        if self.RESIZE_FLAG:
            img = cv2.resize(img, (self.resized_shape[1], self.resized_shape[0]), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return np.array(img_rgb)

    def get_image_name(self, image_id):
        for image in self.images:
            if str(image['id']) == str(image_id):
                return image['file_name']
        raise ValueError(f"No image found with ID {image_id}")

    # def get_image_name(self, image_id):
    #     for image in self.images:
    #         if image['id'] == image_id:
    #             return image['file_name']

    # def get_image_shape(self, image_id):
    #     for image in self.images:
    #         if image['id'] == image_id:
    #             return (image['height'], image['width'])

    def get_image_shape(self, image_id):
        for image in self.images:
            if str(image['id']) == str(image_id):
                return (image['height'], image['width'])
        print(f"No matching image found for image_id: {image_id}")  # Debugging output
        return None

    def draw_with_image_id(self, image_id):
        original_image = self.get_original_image(image_id)
        self.draw_segm_from_coco_annotations(image_id, original_image, self.annotations, True)

    def augment_one_image(self, image_id):
        annotation_ids = []
        for annotation in self.annotations:
            annotation_ids.append(int(annotation['id']))

        max_annotation_id = max(annotation_ids)
        counter = 1
        masks, class_ids = self.get_segm_mask_from_coco_annotations(self.annotations, image_id)
        masks = masks.astype(np.uint8)
        img = self.get_original_image(image_id)
        _, _, n_masks = masks.shape
        image_dict = {}
        for img_dic in self.images:
            if img_dic['id'] == image_id:
                image_dict = deepcopy(img_dic)
                break
        # === flip vertically ===
        img_flipped_vertical = np.flip(img, axis=0)
        open_cv_image = cv2.cvtColor(img_flipped_vertical, cv2.COLOR_RGB2BGR)
        image_id_new = f"{image_id}Vertical"
        cv2.imwrite(filename=f"{self.image_folder_path}/{image_id_new}.png", img=open_cv_image)
        image_dict['id'] = f"{image_id_new}"
        image_dict['file_name'] = f"{image_id_new}.png"
        logger.info(f"image_dic: {image_dict}")
        self.images.append(deepcopy(image_dict))
        for index in range(n_masks):
            mask = masks[:, :, index]
            mask = np.flip(mask, axis=0)
            category_info = {"id": class_ids[index], "is_crowd": False}
            anno = pycococreatortools.create_annotation_info(annotation_id=max_annotation_id + counter,
                                                             image_id=f"{image_id_new}",
                                                             category_info=category_info,
                                                             binary_mask=mask.astype(np.uint8),
                                                             image_size=None,
                                                             tolerance=2,
                                                             bounding_box=None)
            counter += 1
            logger.info(anno)
            self.annotations.append(anno)
        # === Flip horizontally ===
        image_id_new = f"{image_id}Horizontal"
        image_dict['id'] = f"{image_id_new}"
        image_dict['file_name'] = f"{image_id_new}.png"
        logger.info(f"image_dic: {image_dict}")
        self.images.append(deepcopy(image_dict))
        img_flipped_horizontal = np.flip(img, axis=1)
        open_cv_image = cv2.cvtColor(img_flipped_horizontal, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename=f"{self.image_folder_path}/{image_id_new}.png", img=open_cv_image)
        for index in range(n_masks):
            mask = masks[:, :, index]
            mask = np.flip(mask, axis=1)
            category_info = {"id": class_ids[index], "is_crowd": False}
            anno = pycococreatortools.create_annotation_info(annotation_id=max_annotation_id + counter,
                                                             image_id=f"{image_id_new}",
                                                             category_info=category_info,
                                                             binary_mask=mask.astype(np.uint8),
                                                             image_size=None,
                                                             tolerance=2,
                                                             bounding_box=None)
            counter += 1
            logger.info(anno)
            self.annotations.append(anno)
        # === flip both directions ===
        image_id_new = f"{image_id}Both"
        image_dict['id'] = f"{image_id_new}"
        image_dict['file_name'] = f"{image_id_new}.png"
        logger.info(f"image_dic: {image_dict}")
        self.images.append(deepcopy(image_dict))
        img_flipped_both = np.flip(img, axis=(0, 1))
        open_cv_image = cv2.cvtColor(img_flipped_both, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename=f"{self.image_folder_path}/{image_id_new}.png", img=open_cv_image)
        for index in range(n_masks):
            mask = masks[:, :, index]
            mask = np.flip(mask, axis=(0, 1))
            category_info = {"id": class_ids[index], "is_crowd": False}
            anno = pycococreatortools.create_annotation_info(annotation_id=max_annotation_id + counter,
                                                             image_id=f"{image_id_new}",
                                                             category_info=category_info,
                                                             binary_mask=mask.astype(np.uint8),
                                                             image_size=None,
                                                             tolerance=2,
                                                             bounding_box=None)
            counter += 1
            logger.info(anno)
            self.annotations.append(anno)

    def augmentation(self):
        if 'augmented' in self.info:
            logger.debug("Already Augmented")
            pass
        else:
            logger.debug("Start Augmentation")
            for image_id in self.image_ids:
                self.augment_one_image(image_id)
            self.info['augmented'] = True
            with open(self.file, 'w') as f:
                json.dump({
                    "info": self.info,
                    "licenses": self.licenses,
                    "images": self.images,
                    "annotations": self.annotations,
                    "categories": self.categories,
                    "segment_info": self.segment_info
                },
                    f,
                    indent=4)

    def make_train_sample(self, n, file):
        images = []
        annotations = []
        images += self.images[:n]
        for image in images:
            image_id = image['id']
            for anno in self.annotations:
                if anno['image_id'] == image_id:
                    annotations.append(anno)
        anno_json = {
            "info": self.info,
            "licenses": self.licenses,
            "images": images,
            "annotations": annotations,
            "categories": self.categories,
        }
        with open(file, 'w') as f:
            json.dump(anno_json, f)
