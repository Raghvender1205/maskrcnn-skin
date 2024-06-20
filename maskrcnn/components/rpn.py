import tensorflow as tf
from loguru import logger

from maskrcnn.components import Backbone
from maskrcnn.utils import BoundingBoxTools

tf.compat.v1.disable_eager_execution()


class RPN:
    def __init__(self,
                 backbone_model: int = 1,
                 lambda_factor: int = 1,
                 batch: int = 1,
                 learning_rate: float = 1e-4):
        self.LAMBDA_FACTOR = lambda_factor
        self.BATCH = batch
        self.lr = learning_rate
        self.backbone_model = backbone_model
        back_outshape = backbone_model.output.shape[1:]
        self.input_backbone = tf.keras.layers.Input(shape=backbone_model.input.shape[1:], name='backbone_input',
                                                    dtype=tf.float32)
        self.input_rpn = tf.keras.layers.Input(shape=back_outshape, batch_size=None, name='input_rpn', dtype=tf.float32)
        self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                                            kernel_initializer='he_normal', name='rpn_start')(self.input_rpn)
        self.bn1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.act1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bn1)

        # Decide foreground or background
        # [1, 0] is background, [0, 1] is foreground
        self.conv2 = tf.keras.layers.Conv2D(filters=18, kernel_size=(1, 1), padding='same',
                                            kernel_initializer='he_normal')(self.act1)
        self.bn2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.act2 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bn2)
        self.reshape2 = tf.keras.layers.Reshape(target_shape=(back_outshape[0], back_outshape[1], int(18 / 2), 2))(
            self.act2)
        self.rpn_anchor_pred = tf.nn.softmax(logits=self.reshape2, axis=-1, name='rpn_anchor_pred')

        # bounding box regression
        self.conv3 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 1), padding='same',
                                            kernel_initializer='he_normal')(self.act1)
        self.bn3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.act3 = tf.keras.layers.Activation(activation=tf.keras.activations.linear)(self.bn3)
        self.rpn_bbox_regression_pred = tf.keras.layers.Reshape(
            target_shape=(back_outshape[0], back_outshape[1], int(36 / 4), 4), name='rpn_bbox_regression_pred'
        )(self.act3)

        self.rpn_header_model = tf.keras.Model(inputs=[self.input_rpn],
                                               outputs=[self.rpn_anchor_pred, self.rpn_bbox_regression_pred],
                                               name='rpn_backbone_model')
        backbone_out = backbone_model(self.input_backbone)
        rpn_anchor_pred, rpn_bbox_regression_pred = self.rpn_header_model(backbone_out)
        self.rpn_backbone_model = tf.keras.Model(inputs=[self.input_backbone],
                                                 outputs=[rpn_anchor_pred, rpn_bbox_regression_pred],
                                                 name='rpn_backbone_model')
        # print("layer", self.rpn_header_model.outputs)
        self.shape_anchor_target = self.rpn_header_model.get_layer(
            name="tf_op_layer_rpn_anchor_pred" # name='tf.nn.softmax'
        ).get_output_shape_at(0)[1:-1]
        self.shape_bbox_regression = self.rpn_header_model.get_layer(
            name='rpn_bbox_regression_pred'
        ).get_output_shape_at(0)[1:]
        self.n_total_anchors = self.shape_anchor_target[0] * self.shape_anchor_target[1] * self.shape_anchor_target[2]

        # low level training
        self.optimizer_with_backbone = tf.keras.optimizers.Adam(self.lr)
        self.optimizer_header = tf.keras.optimizers.Adam(self.lr)

    def process_image(self, img):
        rpn_anchor_pred, rpn_bbox_regression_pred = self.rpn_backbone_model.predict(img)

        return rpn_anchor_pred, rpn_bbox_regression_pred

    def save_model(self, root_path: str):
        self.rpn_backbone_model.save_weights(filepath=f"{root_path}/rpn_model")

    def load_model(self, root_path: str):
        self.rpn_backbone_model.load_weights(filepath=f"{root_path}/rpn_model")

    def plot_model(self, root_path: str):
        tf.keras.utils.plot_model(self.rpn_header_model, to_file="rpn_header_model.jpg", show_shapes=True)
        tf.keras.utils.plot_model(self.rpn_header_model, to_file="rpn_backbone_model.jpg", show_shapes=True)

        self._rpn_train_model()

    def _rpn_train_model(self):
        self.rpn_anchor_target = tf.keras.layers.Input(shape=self.shape_anchor_target, name='rpn_anchor_target')
        self.rpn_bbox_regression_target = tf.keras.layers.Input(shape=self.shape_bbox_regression,
                                                                name='rpn_bbox_regression_target')
        self.rpn_train_model = tf.keras.Model(
            inputs=[self.rpn_backbone_model.inputs, self.rpn_anchor_target, self.rpn_bbox_regression_target],
            outputs=[self.rpn_backbone_model.outputs], name='rpn_train_model')
        self.rpn_train_model.add_loss(losses=self._rpn_loss(anchor_target=self.rpn_anchor_target,
                                                            bbox_regression_target=self.rpn_bbox_regression_target,
                                                            anchor_pred=self.rpn_backbone_model.outputs[0],
                                                            bbox_regression_pred=self.rpn_backbone_model.outputs[1]))
        self.rpn_train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

        tf.keras.utils.plot_model(model=self.rpn_train_model, to_file='rpn_train_model.jpg', show_shapes=True)

    def _rpn_loss(self, anchor_target, bbox_regression_target, anchor_pred, bbox_regression_pred):
        # input_anchor_target shape: (BS, h, w, n_anchors)
        bbox_inside_weight = tf.zeros(
            shape=(self.BATCH, self.shape_anchor_target[0], self.shape_anchor_target[1], self.shape_anchor_target[2]),
            dtype=tf.float32
        )

        # Anchor Target, 1: foreground, 0.5: ignore, 0: background
        indices_foreground = tf.where(tf.equal(anchor_target, 1))
        indices_background = tf.where(tf.equal(anchor_target, 0))
        logger.info('indices_foreground: {}'.format(indices_foreground))
        logger.info('indices_background: {}'.format(indices_background))
        n_foreground = tf.gather_nd(tf.shape(indices_foreground), [[0]])
        logger.info('n_foreground: {}'.format(n_foreground))

        # update value of bbox_inside_weight corresponding to foreground to 1
        bbox_inside_weight = tf.tensor_scatter_nd_update(tensor=bbox_inside_weight, indices=indices_foreground,
                                                         updates=tf.ones(shape=n_foreground))

        # balance foreground and background training sample
        n_background_selected = 128
        selected_ratio = (n_foreground + n_background_selected) / self.n_total_anchors
        remain_ratio = (self.n_total_anchors - n_foreground - n_background_selected) / self.n_total_anchors
        concat = tf.concat([selected_ratio, remain_ratio], axis=0)
        concat = tf.reshape(concat, (1, 2))

        temp_random_choice = tf.random.categorical(tf.math.log(concat), self.n_total_anchors)
        temp_random_choice = tf.reshape(temp_random_choice, (
            self.BATCH, self.shape_anchor_target[0], self.shape_anchor_target[1], self.shape_anchor_target[2]
        ))
        temp_random_choice = tf.gather_nd(params=temp_random_choice, indices=indices_background)
        temp_random_choice = tf.dtypes.cast(temp_random_choice, tf.float32)

        # update value of bbox_inside_target corresponding to random selected background to 1
        bbox_inside_weight = tf.tensor_scatter_nd_update(tensor=bbox_inside_weight, indices=indices_background,
                                                             updates=temp_random_choice)
        indices_train = tf.where(tf.equal(bbox_inside_weight, 1))
        logger.info(f'Anchor Target: {anchor_target}')
        logger.info(f'Anchor Pred: {anchor_pred}')

        # Train anchor for foreground and background
        anchor_target = tf.cast(anchor_target, tf.int32)
        anchor_target = tf.one_hot(indices=anchor_target, depth=2, axis=-1)
        anchor_target = tf.gather_nd(params=anchor_target, indices=indices_train)
        anchor_pred = tf.gather_nd(params=anchor_pred, indices=indices_train)

        # Train bbox regression only for background
        bbox_regression_target = tf.gather_nd(params=bbox_regression_target, indices=indices_foreground)
        bbox_regression_pred = tf.gather_nd(params=bbox_regression_pred, indices=indices_foreground)

        anchor_loss = tf.losses.categorical_crossentropy(y_true=anchor_target, y_pred=anchor_pred)
        anchor_loss = tf.math.reduce_mean(anchor_loss)
        huber_loss = tf.losses.Huber()
        bbox_regression_loss = huber_loss(y_true=bbox_regression_target, y_pred=bbox_regression_pred)
        bbox_regression_loss = tf.math.reduce_mean(bbox_regression_loss)
        bbox_regression_loss = tf.math.multiply(bbox_regression_loss, self.LAMBDA_FACTOR)
        total_loss = tf.add(anchor_loss, bbox_regression_loss)

        return total_loss

    def _proposal_boxes(self, rpn_anchor_pred,
                        rpn_bbox_regression_pred,
                        anchor_candidates,
                        h: int, w: int, n_anchors: int,
                        n_proposal: int, anchor_threshold: float):
        # Selection part
        rpn_anchor_pred = tf.slice(rpn_anchor_pred, [0, 0, 0, 0, 1], [1, h, w, n_anchors, 1])
        # Squeeze the pred of anchor and bbox_regression
        rpn_anchor_pred = tf.squeeze(rpn_anchor_pred)
        rpn_bbox_regression_pred = tf.squeeze(rpn_bbox_regression_pred)
        shape1 = tf.shape(rpn_anchor_pred)
        # Flatten the pred of anchor to get top N values and indices
        rpn_anchor_pred = tf.reshape(rpn_anchor_pred, (-1,))
        n_anchor_proposal = n_proposal

        top_values, top_indices = tf.nn.top_k(rpn_anchor_pred, n_anchor_proposal)
        top_indices = tf.gather_nd(top_indices, tf.where(tf.greater(top_values, anchor_threshold)))
        top_values = tf.gather_nd(top_values, tf.where(tf.greater(top_values, anchor_threshold)))

        top_indices = tf.reshape(top_indices, (-1, 1))
        update_value = tf.math.add(top_values, 1)
        rpn_anchor_pred = tf.tensor_scatter_nd_update(rpn_anchor_pred, top_indices, update_value)
        rpn_anchor_pred = tf.reshape(rpn_anchor_pred, shape1)

        # Find the base boxes
        anchor_pred_top_indices = tf.where(tf.greater(rpn_anchor_pred, 1))
        base_boxes = tf.gather_nd(anchor_candidates, anchor_pred_top_indices)

        # Find the bbox regressions
        # flatten bbox_regression by last dim to use top_indices to get final_box_regression
        rpn_bbox_regression_pred_shape = tf.shape(rpn_bbox_regression_pred)
        rpn_bbox_regression_pred = tf.reshape(rpn_bbox_regression_pred_shape, (-1, rpn_bbox_regression_pred_shape[-1]))
        final_bbox_regression = tf.gather_nd(rpn_bbox_regression_pred, top_indices)

        # Convert to np to plot
        final_box = BoundingBoxTools.bbox_reg2truebox(base_boxes=base_boxes, regs=final_bbox_regression)

        return final_box

    @tf.function
    def train_step_backbone(self, image, anchor_target, bbox_regression_target):
        with tf.GradientTape() as backbone_tape:
            anchor_pred, bbox_regression_pred = self.rpn_backbone_model(image)
            total_loss = self._rpn_loss(anchor_target, bbox_regression_target, anchor_pred, bbox_regression_pred)
        gradients_backbone = backbone_tape.gradient(total_loss, self.rpn_backbone_model.trainable_variables)
        self.optimizer_with_backbone.apply_gradients(
            zip(gradients_backbone, self.rpn_backbone_model.trainable_variables))

    @tf.function
    def train_step_header(self, image, anchor_target, box_regression_target):
        with tf.GradientTape() as header_tape:
            anchor_pred, bbox_regression_pred = self.rpn_backbone_model(image)
            total_loss = self._rpn_loss(anchor_target, box_regression_target, anchor_pred, bbox_regression_pred)
        gradients_header = header_tape.gradient(total_loss, self.rpn_header_model.trainable_variables)
        self.optimizer_header.apply_gradients(zip(gradients_header, self.rpn_header_model.trainable_variables))