import numpy as np


class FasterRCNNConfig:
    NAME = ""
    N_STAGE = 5
    BATCH_RPN = 1
    BATCH_ROI = 4

    # Anchor Generator
    ANCHOR_BASE_SIZE = 12
    ANCHOR_RATIOS = None
    ANCHOR_SCALES = 2 ** np.arange(3, 6)
    N_ANCHORS = 9

    # Train Generator
    THRESHOLD_IOU_RPN = 0.7
    THRESHOLD_IOU_ROI = 0.55

    # RPN
    PATH_MODEL = "SavedModels"
    PATH_DEBUG_IMG = "SavedDebugImages"
    RPN_LAMBDA_FACTOR = 1 # for balancing RPN losses
    IMG_ORIGINAL_SHAPE = (512, 512, 3)
    IMG_RESIZED_SHAPE = (512, 512, 3)

    ANCHOR_N_PROPOSAL = 300
    ANCHOR_THRESHOLD = 0.51
    RPN_NMS_THRESHOLD = 0.4

    # ROI
    N_OUT_CLASS = 80

    # TRAIN Config
    LR = 1e-4
    EPOCH = 20

    DATA_JSON_FILE = ""
    PATH_IMAGES = ""