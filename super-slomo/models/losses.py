import tensorflow as tf

import config


@tf.function
def reconstruction_loss(y_true, y_pred):
    """
    Reconstruction loss models how good
    the reconstruction of the intermediate frames
    :param y_true: Ground truth values
    :param y_pred: The predicted values
    :return:
    """
    return l1_loss(y_true, y_pred)


@tf.function
def perceptual_loss(vgg16, y_true, y_pred):
    """

    :param vgg16: vgg16 pretrained for perceptual loss
    :param y_true: Ground truth values
    :param y_pred: The predicted values
    :return:
    """
    y_true = vgg16(y_true)
    y_pred = vgg16(y_pred)
    return l2_loss(y_true, y_pred)


@tf.function
def warping_loss(frame_0, frame_t, frame_1, backwarp_frames):
    """
    Warping loss lw to model the quality of the computed optical flow
    :param frame_0:
    :param frame_t:
    :param frame_1:
    :param backwarp_frames:
    :return:
    """
    return (
        l1_loss(frame_0, backwarp_frames[0])
        + l1_loss(frame_1, backwarp_frames[1])
        + l1_loss(frame_t, backwarp_frames[2])
        + l1_loss(frame_t, backwarp_frames[3])
    )


@tf.function
def smoothness_loss(f_01, f_10):
    """
    Smoothness term to encourage neighboring pixels to have similar flow values
    :param f_01: flow from frames_0 to frames_1
    :param f_10: flow from frames_1 to frames_0
    :return: the smoothness loss
    """
    delta_f_01 = _compute_delta(f_01)
    delta_f_10 = _compute_delta(f_10)
    return 0.5 * (delta_f_01 + delta_f_10)


@tf.function
def _compute_delta(frame):
    """

    :param frame:
    :return:
    """
    return tf.reduce_mean(
        tf.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1])
    ) + tf.reduce_mean(tf.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1]))


@tf.function
def l1_loss(y_true, y_pred):
    """
    L1 norm
    :param y_true: Ground truth values
    :param y_pred: The predicted values
    :return: the l1 norm between y_true and y_pred
    """
    return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(y_pred, y_true))))


@tf.function
def l2_loss(y_true, y_pred):
    """
    L2 norm
    :param y_true: Ground truth values
    :param y_pred: The predicted values
    :return: the l2 norm between y_true and y_pred
    """
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y_true)))


@tf.function
def compute_losses(predictions, loss_values, inputs, frames_t, vgg16):
    """
    Compute the losses (reconstruction loss, perceptual loss, smoothness loss, warping loss
    and a comination of all the losses.
    :param predictions: the predictions of the models
    :param loss_values: loss values from the GradientTape
    :param inputs: frames in input
    :param frames_t: target frames
    :param vgg16: vgg16 pretrained for perceptual loss
    :return: the losses
    """
    frames_0, frames_1, _ = inputs
    # unpack loss variables
    f_01, f_10, f_t0, f_t1 = loss_values[:4]
    backwarp_frames = loss_values[4:]
    rec_loss = reconstruction_loss(frames_t, predictions)
    perc_loss = perceptual_loss(vgg16, frames_t, predictions)
    smooth_loss = smoothness_loss(f_01, f_10)
    warp_loss = warping_loss(frames_0, frames_t, frames_1, backwarp_frames)
    total_loss = (
        config.REC_LOSS * rec_loss
        + config.PERCEP_LOSS * perc_loss
        + config.WRAP_LOSS * warp_loss
        + config.SMOOTH_LOSS * smooth_loss
    )
    return total_loss, rec_loss, perc_loss, smooth_loss, warp_loss
