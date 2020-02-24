import tensorflow as tf

import config


class Losses:
    def __init__(self):
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)

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
        y_true = self.vgg16(y_true)
        y_pred = self.vgg16(y_pred)
        return self.mse(y_true, y_pred)

    @tf.function
    def warping_loss(self, frame_0, frame_t, frame_1, backwarp_frames):
        """
        Warping loss lw to model the quality of the computed optical flow
        :param frame_0:
        :param frame_t:
        :param frame_1:
        :param backwarp_frames:
        :return:
        """
        return (
            self.mae(frame_t, backwarp_frames[0])
            + self.mae(frame_t, backwarp_frames[1])
            + self.mae(frame_1, backwarp_frames[2])
            + self.mae(frame_0, backwarp_frames[3])
        )

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
        return tf.reduce_mean(
            tf.abs(frame[:, 1:, :, :] - frame[:, :-1, :, :])
        ) + tf.reduce_mean(tf.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :]))

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

        # unpack loss variables
        for true, pred, loss in zip(frames_t, predictions, loss_values):
            f_01, f_10, f_t0, f_t1 = loss[:4]
            backwarp_frames = loss[4:]
            rec_loss += self.reconstruction_loss(true, pred)
            perc_loss += self.perceptual_loss(true, pred)
            smooth_loss += self.smoothness_loss(f_01, f_10)
            warp_loss += self.warping_loss(frames_0, true, frames_1, backwarp_frames)

        rec_loss /= len(predictions)
        perc_loss /= len(predictions)
        warp_loss /= len(predictions)

        total_loss = (
            config.REC_LOSS * rec_loss
            + config.PERCEP_LOSS * perc_loss
            + config.WRAP_LOSS * warp_loss
            + config.SMOOTH_LOSS * smooth_loss
        )
        return total_loss, rec_loss, perc_loss, smooth_loss, warp_loss
