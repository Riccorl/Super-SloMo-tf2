import argparse
import datetime
import os
import pathlib

import tensorflow as tf
from models import losses, metrics

import config
import dataset
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

    # Tensorboard log directories
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir / "gradient_tape" / current_time / "train"
    test_log_dir = log_dir / "gradient_tape/" / current_time / "test"
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

    # Custom training
    model = SloMoNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, str(chckpnt_dir), max_to_keep=3)
    num_epochs = 1

    if manager.latest_checkpoint:
        status = ckpt.restore(manager.latest_checkpoint).assert_nontrivial_match()
        print("Restored from {}.".format(manager.latest_checkpoint))
        num_epochs = int(manager.latest_checkpoint.split("-")[-1]) + 1
    else:
        print("No checkpoint provided, starting new train.")

    loss_obj = losses.Losses()

    for epoch in range(num_epochs, epochs):
        print("Epoch: " + str(epoch))
        # Average metrics for tensorboard
        avg_losses, avg_metrics = [0] * 5, [0] * 2
        avg_val_losses, avg_val_metrics = [0] * 5, [0] * 2

        for step, frames in enumerate(train_ds):
            inputs, targets = frames
            loss_values, metric_values = train_step(
                model, inputs, targets, optimizer, loss_obj
            )
            avg_losses = [sum(x) for x in zip(avg_losses, loss_values)]
            avg_metrics = [sum(x) for x in zip(avg_losses, avg_metrics)]
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
        avg_losses = [x / len(avg_losses) for x in avg_losses]
        avg_metrics = [x / len(avg_metrics) for x in avg_metrics]
        with train_summary_writer.as_default():
            tf.summary.scalar("total-loss", avg_losses[0], step=epoch)
            tf.summary.scalar("rec_loss", avg_losses[1], step=epoch)
            tf.summary.scalar("perc-loss", avg_losses[2], step=epoch)
            tf.summary.scalar("smooth_loss", avg_losses[3], step=epoch)
            tf.summary.scalar("warping-loss", avg_losses[4], step=epoch)
            tf.summary.scalar("psnr", avg_metrics[0], step=epoch)
            tf.summary.scalar("ssim", avg_metrics[1], step=epoch)

        for step, frames in enumerate(valid_ds):
            inputs, targets = frames
            val_loss_values, val_metric_values = valid_step(
                model, inputs, targets, loss_obj
            )
            avg_val_losses = [sum(x) for x in zip(avg_val_losses, val_loss_values)]
            avg_val_metrics = [sum(x) for x in zip(avg_val_metrics, val_metric_values)]
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

        avg_val_losses = [x / len(avg_val_losses) for x in avg_val_losses]
        #avg_val_metrics = [x / len(avg_val_metrics) for x in avg_val_metrics]

        avg_val_metrics = [
            tf.reduce_mean(x) / len(avg_val_metrics) for x in avg_val_metrics
        ]
       
       
        with test_summary_writer.as_default():
            tf.summary.scalar("val_tot_loss", avg_val_losses[0], step=epoch)
            tf.summary.scalar("val_rec_loss", avg_val_losses[1], step=epoch)
            tf.summary.scalar("val_perc_loss", avg_val_losses[2], step=epoch)
            tf.summary.scalar("val_smooth_loss", avg_val_losses[3], step=epoch)
            tf.summary.scalar("val_warping_loss", avg_val_losses[4], step=epoch)
            tf.summary.scalar("val_psnr", avg_val_metrics[0], step=epoch)
            tf.summary.scalar("val_ssim", avg_val_metrics[1], step=epoch)

        ckpt.step.assign_add(1)
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

    final_file = model_dir / "weights_final_{}.tf".format(epochs)
    model.save_weights(str(final_file), save_format="tf")


@tf.function
def train_step(model, inputs, targets, optimizer, loss_obj):
    """
    Train step for the model in input
    :param model: SloMo model
    :param inputs: frames in input
    :param targets: target frames
    :param optimizer: the optimizer
    :param loss_obj: the loss object
    :return: loss values and metrics values
    """
    with tf.GradientTape() as tape:
        predictions, losses_output = model(inputs, training=True)
        loss_values = loss_obj.compute_losses(
            predictions, losses_output, inputs, targets
        )
        metric_values = metrics.compute_metrics(targets, predictions)

    grads = tape.gradient(loss_values, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_values, metric_values


@tf.function
def valid_step(model, inputs, targets, loss_obj):
    """
    Validation step for the model in input
    :param model: SloMo model
    :param inputs: frames in input
    :param targets: target frames
    :param loss_obj: the loss object
    :return: loss values and metrics values
    """
    predictions, losses_output = model(inputs, training=False)
    loss_values = loss_obj.compute_losses(predictions, losses_output, inputs, targets)
    metric_values = metrics.compute_metrics(targets, predictions)
    return loss_values, metric_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="path to dataset folder", dest="data_dir")
    parser.add_argument("--model", help="path where to save model", required=True)
    parser.add_argument("--epochs", help="number of epochs", default=40, type=int)
    parser.add_argument(
        "--batch_size", help="size of the batch", default=32, type=int,
    )
    parser.add_argument(
        "--frames", help="number of target frames", default=9, type=int
    )
    return parser.parse_args()


def main():
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir / current_time / "train"
    train_log_dir.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    train(args.data_dir, args.model, train_log_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()