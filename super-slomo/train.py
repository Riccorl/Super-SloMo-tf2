import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from models import losses
from models.slomo_model import SloMoNet


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
    print("TensorFlow version: {}".format(tf.version))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    train_dir = Path("../data/extracted/train")
    train_ds = tf.data.Dataset.list_files(str(train_dir / "*"))
    train_ds = train_ds.map(load_frames, num_parallel_calls=12)
    train_ds = prepare_for_training(train_ds)
    image_batch = next(iter(train_ds))

    # Custom training
    model = SloMoNet()
    # Keep results for plotting
    train_loss_results = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    epochs = range(10)
    for epoch in epochs:
        epoch_loss_avg = tf.keras.metrics.Mean()
        with tf.GradientTape() as tape:
            # TODO training, ref:
            # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/customization/custom_training_walkthrough.ipynb
            pass


def main():
    train()


if __name__ == "__main__":
    # main()
    t = np.linspace(0.125, 0.875, 12)
    print(t)
