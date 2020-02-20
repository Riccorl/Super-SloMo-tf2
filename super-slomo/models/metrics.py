import tensorflow as tf


@tf.function
def compute_psnr(frames_t, preds_t):
    """
    Returns the Peak Signal-to-Noise Ratio between the images in frames_t and preds_t.
    :param frames_t: First image batch.
    :param preds_t: Second image batch.
    :return: The scalar PSNR between frames_t and preds_t.
    """
    return tf.image.psnr(frames_t, preds_t, max_val=1.0)


@tf.function
def compute_ssim(frames_t, preds_t):
    """
    Computes SSIM index between the images in frames_t and preds_t.
    :param frames_t: First image batch.
    :param preds_t: Second image batch.
    :return: A tensor containing an SSIM value for each image in batch
    """
    return tf.image.ssim(frames_t, preds_t, max_val=1.0)


@tf.function
def compute_metrics(frames_t, predictions):
    """
    Computes the metrics (psrn, ssim)
    :param frames_t: frames_t in input
    :param predictions: frames_t predicted by the model
    :return: psrn, ssim
    """
    psnr = compute_psnr(frames_t, predictions)
    ssim = compute_ssim(frames_t, predictions)
    return psnr, ssim
