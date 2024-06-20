import tensorflow as tf

from maskrcnn.components.backbone import Backbone


# def test_backbone():
#     img_shape = (512, 512, 3)
#     input1 = tf.keras.layers.Input(shape=img_shape)
#     base_model


def test():
    t1 = Backbone()
    t1.visualize_model()


test()
