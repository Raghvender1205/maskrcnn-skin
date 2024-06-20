import numpy as np


class GenerateBaseAnchors:
    @classmethod
    def generate_base_anchors(cls, base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = cls._ratio_enum(base_anchor, ratios)
        anchors = np.vstack([cls._scale_enum(ratio_anchors[i, :], scales)
                            for i in range(ratio_anchors.shape[0])])

        return anchors

    @classmethod
    def _whctrs(cls, anchor):
        """
        Return width, height, x_center, y_center for an anchors (window)
        """
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)

        return w, h, x_ctr, y_ctr

    @classmethod
    def _mkanchors(cls, ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr) output a set of anchors (windows)
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        
        return anchors

    @classmethod
    def _ratio_enum(cls, anchor, ratios):
        """
        Enumerate a set of anchors for each aspect ratio w.r.t an anchor
        """
        w, h, x_ctr, y_ctr = cls._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = cls._mkanchors(ws, hs, x_ctr, y_ctr)

        return anchors

    @classmethod
    def _scale_enum(cls, anchor, scales):
        """
        Enumerate a set of anchors for each scale w.r.t an anchor
        """
        w, h, x_ctr, y_ctr = cls._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = cls._mkanchors(ws, hs, x_ctr, y_ctr)

        return anchors