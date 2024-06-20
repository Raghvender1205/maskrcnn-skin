import tensorflow as tf


class Backbone:
    def __init__(self, img_shape: tuple = (512, 512, 3), n_stage: int = 5):
        # ResNet50V2 has 5 stages
        self.base_model = tf.keras.applications.ResNet50V2(input_shape=img_shape, include_top=False, weights='imagenet')

        if n_stage == 4:
            self.backbone_model = tf.keras.Model(inputs=[self.base_model.input], outputs=[
                self.base_model.get_layer(name='conv4_block5_preact_relu').output], name='BACKBONE_MODEL')
        elif n_stage == 5:
            self.backbone_model = tf.keras.Model(inputs=[self.base_model.input], outputs=[
                self.base_model.output], name='BACKBONE_MODEL')

    def visualize_model(self):
        tf.keras.utils.plot_model(self.backbone_model, to_file='backbone_model_modified.png', show_shapes=True)

    def get_output_shape(self):
        return self.backbone_model.layers[-1].output_shape[1:]

    def save_weight(self, root_path):
        self.backbone_model.save_weights(filepath=f'{root_path}/backbone_model', overwrite=True)

    def load_weight(self, root_path):
        self.backbone_model.load_weights(filepath=f'{root_path}/backbone_model')