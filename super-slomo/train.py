import pathlib
import random

import tensorflow as tf

from models import losses
from models.slomo_model import SloMoNet
import config
import os
import datetime


def prepare_for_training(dataset, batch_size=32, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def load_frames(folder_path):
    files = tf.io.matching_files(folder_path + "/*.jpg")
    sampled_files = [files[i] for i in sorted(random.sample(range(12), 3))]
    # load the raw data from the file as a string
    decoded = [decode_img(tf.io.read_file(f)) for f in sampled_files]
    return decoded


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [256, 256])


def train(log_dir):
    tf.keras.backend.clear_session()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    batch_size = 8
    train_dir = config.TRAIN_DIR
    train_ds = tf.data.Dataset.list_files(str(train_dir / "*"))
    train_ds = (
        train_ds.map(load_frames, num_parallel_calls=12)
        .batch(batch_size)
        .prefetch(buffer_size=batch_size)
    )

    # Custom training
    model = SloMoNet()
    vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)

    # Keep results for plotting
    train_loss_results = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    epochs = range(10)
    for epoch in epochs:
        print("Epoch: " + str(epoch))
        epoch_total_loss_avg = tf.keras.metrics.Mean()
        epoch_rec_loss_avg = tf.keras.metrics.Mean()
        epoch_perc_loss_avg = tf.keras.metrics.Mean()
        epoch_smooth_loss_avg = tf.keras.metrics.Mean()
        epoch_warping_loss_avg = tf.keras.metrics.Mean()
        for frames in train_ds:
            loss_values, grads = grad(model, frames, vgg16)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Track progress
            total_loss, rec_loss, perc_loss, smooth_loss, warping_loss = loss_values
            epoch_total_loss_avg(total_loss)  # Add current batch loss
            epoch_rec_loss_avg(rec_loss)  # Add current batch loss
            epoch_perc_loss_avg(perc_loss)  # Add current batch loss
            epoch_smooth_loss_avg(smooth_loss)  # Add current batch loss
            epoch_warping_loss_avg(warping_loss)  # Add current batch loss
            print(
                "Step total loss: {:.3f}, rec loss: {:.3f}, perc loss: {:.3f}, smooth loss: {:.3f}, warping loss: {:.3f}".format(
                    total_loss, rec_loss, perc_loss, smooth_loss, warping_loss
                )
            )
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', loss_value, step=epoch)

        train_loss_results.append(epoch_rec_loss_avg.result())
        if epoch % 50 == 0:
            print(
                "Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_rec_loss_avg.result())
            )


@tf.function
def grad(model, inputs, vgg16):
    with tf.GradientTape() as tape:
        loss_values = compute_losses(model, inputs, vgg16, training=True)
    return loss_values, tape.gradient(loss_values, model.trainable_variables)


@tf.function
def compute_losses(model, inputs, vgg16, training):
    frame_0, frame_t, frame_1 = inputs
    predictions, losses_output = model(inputs, training=training)
    # unpack loss variables
    f_01, f_10, f_t0, f_t1 = losses_output[:4]
    backwarp_frames = losses_output[4:]
    rec_loss = losses.reconstruction_loss(frame_t, predictions)
    perc_loss = losses.perceptual_loss(vgg16, frame_t, predictions)
    smooth_loss = losses.smoothness_loss(f_01, f_10)
    warping_loss = losses.warping_loss(frame_0, frame_t, frame_1, backwarp_frames)
    total_loss = (
        config.REC_LOSS * rec_loss
        + config.PERCEP_LOSS * perc_loss
        + config.WRAP_LOSS * warping_loss
        + config.SMOOTH_LOSS * smooth_loss
    )
    return total_loss, rec_loss, perc_loss, smooth_loss, warping_loss


def main():
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_dir / current_time / "train"
    train_log_dir.mkdir(parents=True, exist_ok=True)
    train(train_log_dir)


if __name__ == "__main__":
    main()
    # t = np.linspace(0.125, 0.875, 12)
    # print(t)
