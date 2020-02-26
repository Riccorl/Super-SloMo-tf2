import tensorflow as tf

import config


class Losses:
    def __init__(self):
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        model = tf.keras.applications.VGG16(
            weights="imagenet", include_top=False, input_shape=(352, 352, 3)
        )
        self.vgg16 = tf.keras.Model(
            model.inputs, model.get_layer("block4_conv3").output
        )

    @tf.function
    def reconstruction_loss(self, y_true, y_pred):
        """
        Reconstruction loss models how good
        the reconstruction of the intermediate frames
        :param y_true: Ground truth values
        :param y_pred: The predicted values
        :return:
        """
        return self.mae(y_true, y_pred)

    @tf.function
    def perceptual_loss(self, y_true, y_pred):
        """

        :param y_true: Ground truth values
        :param y_pred: The predicted values
        :return:
        """
        # y_true = self.vgg16.predict((tf.expand_dims(y_true, 0)), steps=1)
        # y_pred = self.vgg16.predict((tf.expand_dims(y_pred, 0)), steps=1)
        y_true = self.extract_feat(self.vgg16, tf.expand_dims(y_true, 0))
        y_pred = self.extract_feat(self.vgg16, tf.expand_dims(y_pred, 0))
        return self.mse(y_true, y_pred)

    @tf.function
    def extract_feat(self, feat_extractor, inputs):
        """

        :param feat_extractor:
        :param inputs:
        :return:
        """
        feats = inputs
        for layer in feat_extractor.layers:
            feats = layer(feats)
        return feats

    @tf.function
    def warping_loss(self, frame_0, frames_t, frame_1, bw_values, batch, n):
        """
        Warping loss lw to model the quality of the computed optical flow
        :param frame_0:
        :param frames_t:
        :param frame_1:
        :param bw_values:
        :return:
        """
        w_1 = self.mae(frame_0[batch], bw_values[0][batch])
        w_2 = self.mae(frame_1[batch], bw_values[1][batch])
        w_3 = sum([self.mae(frames_t[i], bw_values[2][batch][i]) for i in range(n)]) / n
        w_4 = sum([self.mae(frames_t[i], bw_values[3][batch][i]) for i in range(n)]) / n
        return w_1 + w_2 + w_3 + w_4

    @tf.function
    def smoothness_loss(self, f_01, f_10):
        """
        Smoothness term to encourage neighboring pixels to have similar flow values
        :param f_01: flow from frames_0 to frames_1
        :param f_10: flow from frames_1 to frames_0
        :return: the smoothness loss
        """
        delta_f_01 = self._compute_delta(f_01)
        delta_f_10 = self._compute_delta(f_10)
        return delta_f_01 + delta_f_10

    @tf.function
    def _compute_delta(self, frame):
        """

        :param frame:
        :return:
        """
        x = tf.reduce_mean(tf.abs(frame[1:, :, :] - frame[:-1, :, :]))
        y = tf.reduce_mean(tf.abs(frame[:, 1:, :] - frame[:, :-1, :]))
        return x + y

    def compute_losses(self, predictions, loss_values, inputs, frames_t):
        """
        Compute the losses (reconstruction loss, perceptual loss, smoothness loss, warping loss
        and a comination of all the losses.
        :param predictions: the predictions of the models
        :param loss_values: loss values from the GradientTape
        :param inputs: frames in input
        :param frames_t: target frames
        :return: the losses
        """
        rec_loss, perc_loss, warp_loss, smooth_loss = 0, 0, 0, 0
        frames_0, frames_1, _ = inputs
        n_frames = frames_t.shape[1]

        # unpack loss variables
        f_01, f_10 = loss_values[:2]
        backwarp_values = loss_values[2:]

        for i, (batch_true, batch_pred) in enumerate(zip(frames_t, predictions)):
            p_rec_loss, p_perc_loss = 0, 0
            for true, pred in zip(batch_true, batch_pred):
                p_rec_loss += self.reconstruction_loss(true, pred)
                p_perc_loss += self.perceptual_loss(true, pred)

            rec_loss += p_rec_loss / n_frames
            perc_loss += p_perc_loss / n_frames
            smooth_loss = self.smoothness_loss(f_01[i], f_10[i])
            warp_loss += self.warping_loss(
                frames_0, batch_true, frames_1, backwarp_values, i, n_frames
            )

        total_loss = (
            config.REC_LOSS * rec_loss
            + config.PERCEP_LOSS * perc_loss
            + config.WRAP_LOSS * warp_loss
            + config.SMOOTH_LOSS * smooth_loss
        )
        return total_loss, rec_loss, perc_loss, smooth_loss, warp_loss
