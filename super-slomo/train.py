import random

import tensorflow as tf

from models import losses
from models.slomo_model import SloMoNet
import config
import os


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
    return tf.image.resize(img, [352, 352])


def train():
    tf.keras.backend.clear_session()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    train_dir = config.TRAIN_DIR
    train_ds = tf.data.Dataset.list_files(str(train_dir / "*"))
    train_ds = (
        train_ds.map(load_frames, num_parallel_calls=12)
        .batch(6)
        .prefetch(buffer_size=6)
    )

    # Custom training
    model = SloMoNet()
    # Keep results for plotting
    train_loss_results = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    epochs = range(10)
    for epoch in epochs:
        epoch_loss_avg = tf.keras.metrics.Mean()
        with tf.GradientTape() as tape:
            for frames in train_ds:
                loss_value, grads = grad(model, frames)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # Track progress
                epoch_loss_avg(loss_value)  # Add current batch loss

        train_loss_results.append(epoch_loss_avg.result())
        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))


def grad(model, inputs):
    with tf.GradientTape() as tape:
        loss_value = compute_losses(model, inputs, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_losses(model, inputs, training):
    frame_0, frame_t, frame_1 = inputs
    predictions, warping_output = model(inputs, training=training)
    rec_loss = losses.reconstruction_loss(frame_t, predictions)
    return rec_loss


def main():
    train()


if __name__ == "__main__":
    main()
    # t = np.linspace(0.125, 0.875, 12)
    # print(t)