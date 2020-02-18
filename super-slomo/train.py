import datetime
import os
import pathlib
import random

import numpy
import tensorflow as tf

import config
from models import losses, metrics
from models.slomo_model import SloMoNet


def load_dataset(
    data_dir: pathlib.Path,
    batch_size: int = 32,
    cache: bool = True,
    train: bool = True,
):
    """
    Prepare the tf.data.Dataset for training
    :param data_dir: directory of the dataset
    :param batch_size: size of the batch
    :param cache: if True, cache the dataset
    :param train: if True, agument and shuffle the dataset
    :return: the dataset in input
    """
    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.list_files(str(data_dir / "*"))
    dataset = dataset.map(load_frames, num_parallel_calls=autotune)
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()
    if train:
        dataset = dataset.map(data_augment, num_parallel_calls=autotune)
        dataset = dataset.shuffle(buffer_size=128)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    dataset = dataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def data_augment(frame_0, frame_t, frame_1, frame_t_index):
    """
    Augment the images in the dataset
    :param frame_0: frame_0
    :param frame_t: frame_t
    :param frame_1: frame_1
    :param frame_t_index: index of frame_t
    :return: the frames augmented
    """
    w, h = 256, 256
    # frame_0, frame_t, frame_1 = frames
    frame_0 = tf.image.resize(frame_0, [w, h])
    frame_t = tf.image.resize(frame_t, [w, h])
    frame_1 = tf.image.resize(frame_1, [w, h])
    # for frame in frames:
    #     tf.image.resize(frame, [w, h])
    # print(frames)
    return frame_0, frame_t, frame_1, frame_t_index


def load_frames(folder_path: str):
    """
    Load the frames in the folder specified by folder_path
    :param folder_path: folder path where frames are located
    :return: the decoded frames
    """
    files = tf.io.matching_files(folder_path + "/*.jpg")
    sampled_indeces = sorted(random.sample(range(12), 3))
    sampled_files = [files[i] for i in sampled_indeces]
    # load the raw data from the file as a string
    decoded = [decode_img(tf.io.read_file(f)) for f in sampled_files]
    return decoded + sampled_indeces[1:2]


def decode_img(image: str):
    """
    Decode the image from its filename
    :param image: the image to decode
    :return: the image decoded
    """
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to the desired size.
    return image


def train(log_dir: pathlib.Path, epochs: int, batch_size: int):
    """
    Train funtion
    :param log_dir: directory where to store logs for Tensorboard
    :param epochs: number of epochs
    :param batch_size: size of the batch
    :return:
    """
    tf.keras.backend.clear_session()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    train_ds = load_dataset(config.TRAIN_DIR, batch_size)
    len_train = len([f.name for f in os.scandir(config.TRAIN_DIR) if f.is_dir()])
    progbar = tf.keras.utils.Progbar(len_train // batch_size)
    valid_ds = load_dataset(config.VALID_DIR, batch_size, train=False)
    val_progbar = tf.keras.utils.Progbar(100 // batch_size)

    # Custom training
    model = SloMoNet(batch_size)
    vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)

    # Keep results for plotting
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        # losses_avg = [tf.keras.metrics.Mean()] * 5
        for step, frames in enumerate(train_ds):
            loss_values, metric_values = train_step(model, frames, optimizer, vgg16)
            progbar.update(
                step + 1,
                [
                    ("total_loss", loss_values[0]),
                    ("rec_loss", loss_values[1]),
                    ("perc_loss", loss_values[2]),
                    ("smooth_loss", loss_values[3]),
                    ("warping_loss", loss_values[4]),
                    ("psnr", metric_values[0]),
                    ("ssim", metric_values[1]),
                ],
            )
        for step, frames in enumerate(valid_ds):
            val_loss_values, val_metric_values = valid_step(model, frames, vgg16)
            val_progbar.update(
                step + 1,
                [
                    ("total_loss", val_loss_values[0]),
                    ("rec_loss", val_loss_values[1]),
                    ("perc_loss", val_loss_values[2]),
                    ("smooth_loss", val_loss_values[3]),
                    ("warping_loss", val_loss_values[4]),
                    ("psnr", val_metric_values[0]),
                    ("ssim", val_metric_values[1]),
                ],
            )


@tf.function
def train_step(model, inputs, optimizer, vgg16):
    """
    Train step for the model in input
    :param model: SloMo model
    :param inputs: frames in input
    :param optimizer: the optimizer
    :param vgg16: vgg16 pretrained for perceptual loss
    :return: loss values and metrics values
    """
    with tf.GradientTape() as tape:
        predictions, losses_output = model(inputs, training=True)
        loss_values = compute_losses(predictions, losses_output, inputs, vgg16)
        metric_values = compute_metrics(inputs[1], predictions)

    grads = tape.gradient(loss_values, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_values, metric_values


@tf.function
def valid_step(model, inputs, vgg16):
    """
    Validation step for the model in input
    :param model: SloMo model
    :param inputs: frames in input
    :param vgg16: vgg16 pretrained for perceptual loss
    :return: loss values and metrics values
    """
    predictions, losses_output = model(inputs, training=False)
    loss_values = compute_losses(predictions, losses_output, inputs, vgg16)
    metric_values = compute_metrics(inputs[1], predictions)
    return loss_values, metric_values


@tf.function
def compute_losses(predictions, loss_values, inputs, vgg16):
    """
    Compute the losses (reconstruction loss, perceptual loss, smoothness loss, warping loss
    and a comination of all the losses.
    :param predictions: the predictions of the models
    :param loss_values: loss values from the GradientTape
    :param inputs: frames in input
    :param vgg16: vgg16 pretrained for perceptual loss
    :return: the losses
    """
    frames_0, frames_t, frames_1, _ = inputs
    # unpack loss variables
    f_01, f_10, f_t0, f_t1 = loss_values[:4]
    backwarp_frames = loss_values[4:]
    rec_loss = losses.reconstruction_loss(frames_t, predictions)
    perc_loss = losses.perceptual_loss(vgg16, frames_t, predictions)
    smooth_loss = losses.smoothness_loss(f_01, f_10)
    warping_loss = losses.warping_loss(frames_0, frames_t, frames_1, backwarp_frames)
    total_loss = (
        config.REC_LOSS * rec_loss
        + config.PERCEP_LOSS * perc_loss
        + config.WRAP_LOSS * warping_loss
        + config.SMOOTH_LOSS * smooth_loss
    )
    return total_loss, rec_loss, perc_loss, smooth_loss, warping_loss


@tf.function
def compute_metrics(frames_t, predictions):
    """
    Computes the metrics (psrn, ssim)
    :param frames_t: frames_t in input
    :param predictions: frames_t predicted by the model
    :return: psrn, ssim
    """
    psnr = metrics.compute_psnr(frames_t, predictions)
    ssim = metrics.compute_ssim(frames_t, predictions)
    return psnr, ssim


def main():
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir / current_time / "train"
    train_log_dir.mkdir(parents=True, exist_ok=True)

    epochs = 10
    batch_size = 6

    train(train_log_dir, epochs, batch_size)


if __name__ == "__main__":
    main()
    # t = numpy.linspace(0, 1, 12)
    # print(t)
    # print(t.shape)
    # tt = numpy.array([t] * 6)
    # print(tt.shape)
