from maskrcnn.utils.cocotools import COCOTools


def test_cocotools():
    file = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/annotations.json"
    file_sample = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/annotations.json"
    image_path = "/Users/raghvender/Desktop/AI/SkinSegmentation/custom_data/train/images"

    t1 = COCOTools(file, image_path, resized_shape=(512, 512, 3))
    t1.make_train_sample(n=20, file=file_sample)


test_cocotools()