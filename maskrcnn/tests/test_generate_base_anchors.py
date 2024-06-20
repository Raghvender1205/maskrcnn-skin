import time
from maskrcnn.utils.generate_base_anchors import GenerateBaseAnchors


t = time.time()
a = GenerateBaseAnchors.generate_base_anchors()
print(time.time() - t)
print(a)
print(a.shape)