from maskrcnn.components import Backbone, RPN


def test_rpn():
    b1 = Backbone()
    t1 = RPN(b1.backbone_model)
    t1.plot_model(root_path="model_images/")


if __name__ == '__main__':
    test_rpn()