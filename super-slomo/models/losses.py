import tensorflow as tf


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
