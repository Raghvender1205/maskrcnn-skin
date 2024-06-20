import numpy as np

from loguru import logger
from maskrcnn.utils.generate_candidate_anchors import GenerateCandidateAnchors


def test_generate_candidate_anchors():
    t1 = GenerateCandidateAnchors(base_size=12, ratios=[0.5, 1, 2])
    logger.debug(f"base anchors {t1.base_anchors}")
    logger.debug(f"9 Anchor candidates at [0, 0]: {t1.anchor_candidates[0, 0, :, :]}")
    logger.debug(f"9 Anchor candidates at [10, 10]: {t1.anchor_candidates[10, 10, :, :]}")
    bboxes = t1.anchor_candidates[10, 10, :, :].tolist()
    t1._validate_bbox(bboxes)
    logger.debug(f"1 Anchor Candidate at [0, 0, 2]: {t1.anchor_candidates[0, 0, 2, :]}")

    temp = np.reshape(t1.anchor_candidates, newshape=(-1, 4))
    logger.debug(f"Same with Anchor Candidate at [0, 0, 2] after reshape: {temp[2, :]}")

    temp2 = temp.reshape((t1.h, t1.w, t1.n_anchors, 4))
    logger.debug(f"Same with temp[2, :] after reshape: {temp2[0, 0, 2, :]}")
    logger.debug(f"Same anchor in anchor list {t1.anchor_candidates_list[2]}")


if __name__ == "__main__":
    test_generate_candidate_anchors()
