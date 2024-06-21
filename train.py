from maskrcnn.model.skin_fasterrcnn import FasterRCNNModel

f1 = FasterRCNNModel()
f1.train_rpn_roi()
f1.save_weight()