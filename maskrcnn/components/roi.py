import tensorflow as tf


class RoI:
    def __init__(self, backbone_model, img_shape, n_output_classes: int = 80, learning_rate: float = 1e-4):
        self.backbone_model = backbone_model
        self.lr = learning_rate
        self.input_backbone = tf.keras.Input(shape=backbone_model.input_shape[1:], dtype=tf.float32,
                                             name='backbone_input')
        proposal_boxes = tf.keras.Input(shape=(4,), batch_size=None, name='proposal_boxes', dtype=tf.float32)
        feature_map_shape = self.backbone_model.layers[-1].output_shape[1:]
        feature_map = tf.keras.layers.Input(shape=feature_map_shape, batch_size=None, name='feature_map',
                                            dtype=tf.float32)

        shape1 = tf.shape(proposal_boxes, out_type=tf.int32)
        n_boxes = tf.gather_nd(shape1, [0])
        indices = tf.zeros(shape=n_boxes, dtype=tf.int32)  # only input 1 image, all indices are 0
        img_shape_constant = tf.constant([img_shape[0], img_shape[1], img_shape[0], img_shape[1]], dtype=tf.float32)
        proposal_boxes2 = tf.math.divide(proposal_boxes, img_shape_constant)

        image_crop = tf.image.crop_and_resize(image=feature_map, boxes=proposal_boxes2,
                                              box_indices=indices, crop_size=[7, 7])

        flatten1 = tf.keras.layers.GlobalAveragePooling2D()(image_crop)
        fc1 = tf.keras.layers.Dense(units=1024, activation='relu')(flatten1)
        class_header = tf.keras.layers.Dense(units=n_output_classes + 1, activation='softmax')(fc1)
        box_regression_header = tf.keras.layers.Dense(units=4, activation='linear')(fc1)

        self.roi_header_model = tf.keras.Model(inputs=[feature_map, proposal_boxes],
                                               outputs=[class_header, box_regression_header],
                                               name='roi_header_model')
        backbone_out = self.backbone_model(self.input_backbone)
        roi_backbone_out1, roi_backbone_out2 = self.roi_header_model([backbone_out, proposal_boxes])

        self.roi_backbone_model = tf.keras.Model(inputs=[self.input_backbone, proposal_boxes],
                                                 outputs=[roi_backbone_out1, roi_backbone_out2])

        self.huber = tf.keras.losses.Huber()
        self.optimizer_backbone = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_header = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def save_header(self, root_path: str):
        self.roi_header_model.save_weights(filepath=f'{root_path}/roi_header_model')

    def load_header(self, root_path: str):
        self.roi_header_model.load_weights(filepath=f'{root_path}/roi_header_model')

    def process_image(self, input_img_box):
        pred_class, pred_box_regression = self.roi_backbone_model(input_img_box)

        return pred_class, pred_box_regression

    def plot_model(self):
        tf.keras.utils.plot_model(self.roi_header_model, 'roi_header_model.jpg', show_shapes=True)
        tf.keras.utils.plot_model(self.roi_backbone_model, 'roi_backbone_model.jpg', show_shapes=True)

    @tf.function
    def train_step_backbone(self, input_image, proposal_box, class_header, box_regression_header):
        with tf.GradientTape() as roi_tape:
            pred_class, box_regression_pred = self.roi_backbone_model([input_image, proposal_box])
            class_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=class_header, y_pred=pred_class)
            box_regression_loss = self.huber(y_true=box_regression_header, y_pred=box_regression_pred)

            total_loss = tf.reduce_mean(tf.add(class_loss, box_regression_loss))
        gradients = roi_tape.gradient(total_loss, self.roi_header_model.trainable_variables)
        self.optimizer_header.apply_gradients(zip(gradients, self.roi_header_model.trainable_variables))

    @tf.function
    def train_step_header(self, input_image, proposal_box, class_header, box_regression_header):
        with tf.GradientTape() as roi_tape:
            class_pred, box_regression_pred = self.roi_backbone_model([input_image, proposal_box])
            class_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=class_header, y_pred=class_pred)

            box_regression_loss = self.huber(y_true=box_regression_header, y_pred=box_regression_pred)
            total_loss = tf.reduce_mean(tf.add(class_loss, box_regression_loss))
        gradients = roi_tape.gradient(total_loss, self.roi_header_model.trainable_variables)
        self.optimizer_header.apply_gradients(zip(gradients, self.roi_header_model.trainable_variables))
