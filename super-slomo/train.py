import argparse
import datetime
import os
import pathlib

import numpy as np
import tensorflow as tf

import config
import dataset
from models import losses, metrics
from models.slomo_model import SloMoNet


def train(
    data_dir: str, model_dir: str, log_dir: pathlib.Path, epochs: int, batch_size: int
):
    """
    Train funtion
    :param data_dir: dataset directory
    :param model_dir: directory where to save the mdoels
    :param log_dir: directory where to store logs for Tensorboard
    :param epochs: number of epochs
    :param batch_size: size of the batch
    :return:
    """
    tf.keras.backend.clear_session()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    data_dir = pathlib.Path(data_dir)
    train_ds = dataset.load_dataset(data_dir / "train", batch_size)
    len_train = tf.data.experimental.cardinality(train_ds).numpy()
    progbar = tf.keras.utils.Progbar(len_train)
    valid_ds = dataset.load_dataset(data_dir / "val", batch_size, train=False)
    len_valid = tf.data.experimental.cardinality(valid_ds).numpy()
    val_progbar = tf.keras.utils.Progbar(len_valid)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    chckpnt_dir = model_dir / "chckpnt"
    chckpnt_dir.mkdir(parents=True, exist_ok=True)

    # Custom training
    model = SloMoNet()
    vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, str(chckpnt_dir), max_to_keep=3)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        for step, frames in enumerate(train_ds):
            inputs, targets = frames
            loss_values, metric_values = train_step(
                model, inputs, targets, optimizer, vgg16
            )
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
            inputs, targets = frames
            val_loss_values, val_metric_values = valid_step(
                model, inputs, targets, vgg16
            )
            val_progbar.update(
                step + 1,
                [
                    ("total_loss", val_loss_values[0]),
                    ("rec_loss", val_loss_values[1]),
                    ("perc_loss", val_loss_values[2]),
                    ("smooth_loss", val_loss_values[3]),
                    ("warping_loss", val_loss_values[4]),
                    ("val_psnr", val_metric_values[0]),
                    ("val_ssim", val_metric_values[1]),
                ],
            )
        ckpt.step.assign_add(1)
        save_path = manager.save()
        # model.save_weights(str(model_dir / "weights_{}.tf".format(epoch)), save_format="tf")
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

    final_file = model_dir / "weights_final_{}.tf".format(epochs)
    model.save_weights(str(final_file), save_format="tf")


@tf.function
def train_step(model, inputs, targets, optimizer, vgg16):
    """
    Train step for the model in input
    :param model: SloMo model
    :param inputs: frames in input
    :param targets: target frames
    :param optimizer: the optimizer
    :param vgg16: vgg16 pretrained for perceptual loss
    :return: loss values and metrics values
    """
    with tf.GradientTape() as tape:
        predictions, losses_output = model(inputs, training=True)
        loss_values = losses.compute_losses(predictions, losses_output, inputs, targets, vgg16)
        metric_values = metrics.compute_metrics(inputs[1], predictions)

    grads = tape.gradient(loss_values, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_values, metric_values


@tf.function
def valid_step(model, inputs, targets, vgg16):
    """
    Validation step for the model in input
    :param model: SloMo model
    :param inputs: frames in input
    :param targets: target frames
    :param vgg16: vgg16 pretrained for perceptual loss
    :return: loss values and metrics values
    """
    predictions, losses_output = model(inputs, training=False)
    loss_values = losses.compute_losses(predictions, losses_output, inputs, targets, vgg16)
    metric_values = metrics.compute_metrics(inputs[1], predictions)
    return loss_values, metric_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="path to dataset folder", dest="data_dir")
    parser.add_argument(help="path where to save model", dest="model_dir")
    parser.add_argument(
        "--epochs", help="number of epochs", dest="epochs", default=40, type=int
    )
    parser.add_argument(
        "--batch-size",
        help="size of the batch",
        dest="batch_size",
        default=32,
        type=int,
    )
    return parser.parse_args()


def main():
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir / current_time / "train"
    train_log_dir.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    train(args.data_dir, args.model_dir, train_log_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
