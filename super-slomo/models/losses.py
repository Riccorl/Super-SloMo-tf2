import tensorflow as tf


@tf.function
def reconstruction_loss(y_true, y_pred):
    return l1_loss(y_true, y_pred)


@tf.function
def perceptual_loss(vgg16, y_true, y_pred):
    y_true = vgg16(y_true)
    y_pred = vgg16(y_pred)
    return l2_loss(y_true, y_pred)


@tf.function
def l1_loss(y_true, y_pred):
    """
    L1 norm
    """
    return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(y_pred, y_true))))


@tf.function
def l2_loss(y_true, y_pred):
    """
    L2 norm
    """
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y_true)))


@tf.function
def warping_loss(frame_0, frame_t, frame_1, backwarp_frames):
    return (
        l1_loss(frame_0, backwarp_frames[0])
        + l1_loss(frame_1, backwarp_frames[1])
        + l1_loss(frame_t, backwarp_frames[2])
        + l1_loss(frame_t, backwarp_frames[3])
    )


@tf.function
def smoothness_loss(f_01, f_10):
    delta_f_01 = _compute_delta(f_01)
    delta_f_10 = _compute_delta(f_10)
    return 0.5 * (delta_f_01 + delta_f_10)


@tf.function
def _compute_delta(frame):
    return tf.reduce_mean(
        tf.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1])
    ) + tf.reduce_mean(tf.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1]))
