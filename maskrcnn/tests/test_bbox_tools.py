import numpy as np

from maskrcnn.utils.bbox_tools import BoundingBoxTools

image_shape = (512, 512) # (h, w)
bbox1_xyxy = np.array([[0, 0, 9, 9]])  # (x,y,x,y)
bbox1_whc = np.array([[5, 5, 10, 10], [5, 5, 10, 10]])
bbox2_xyxy = np.array([[5, 5, 14, 14]])
bbox2_whc = np.array([[10, 10, 10, 10]])

print(BoundingBoxTools.xxyy2xywh(bbox1_xyxy))
print(BoundingBoxTools.xywh2xxyy(bbox1_whc))
print(BoundingBoxTools.ious([[0, 0, 9, 9], [0, 0, 9, 9]], [5, 5, 14, 14]))
print(BoundingBoxTools.bbox_regression_target(bbox1_xyxy, bbox2_xyxy))