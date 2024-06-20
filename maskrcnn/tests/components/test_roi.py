from maskrcnn.components import Backbone, RoI


def test_roi():
    b1 = Backbone()
    t1 = RoI(b1.backbone_model, img_shape=(512, 512, 3))
    t1.plot_model()


if __name__ == '__main__':
    test_roi()