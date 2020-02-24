import pathlib

import tensorflow as tf


def load_dataset(
    data_dir: pathlib.Path,
    batch_size: int = 32,
    buffer_size: int = 1000,
    cache: bool = False,
    train: bool = True,
):
    """
    Prepare the tf.data.Dataset for training
    :param data_dir: directory of the dataset
    :param batch_size: size of the batch
    :param buffer_size: the number of elements from this
        dataset from which the new dataset will sample.
    :param cache: if True, cache the dataset
    :param train: if True, agument and shuffle the dataset
    :return: the dataset in input
    """
    autotune = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.list_files(str(data_dir / "*"))
    ds = ds.map(lambda x: load_frames(x, train), num_parallel_calls=autotune)
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory. It cause memory leak, check with more memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if train:
        ds = ds.shuffle(buffer_size=buffer_size)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(autotune)
    return ds


def load_frames(folder_path: str, train: bool):
    """
    Load the frames in the folder specified by folder_path
    :param folder_path: folder path where frames are located
    :param train: if true, augment images
    :return: the decoded frames
    """
    files = tf.io.matching_files(folder_path + "/*.jpg")

    sampled_indeces = tf.random.shuffle(tf.range(12))[:3]
    sampled_indeces = tf.sort(sampled_indeces)
    sampled_files = tf.gather(files, sampled_indeces)

    frame_0 = decode_img(sampled_files[0])
    frame_1 = decode_img(sampled_files[2])
    frame_t = decode_img(sampled_files[1])
    
    if train:
        frames = data_augment(tf.concat([frame_0, frame_1, frame_t], axis=2))
        frame_0, frame_1, frame_t = frames[:, :, :3], frames[:, :, 3:6], frames[:, :, 6:9]
    return (frame_0, frame_1, sampled_indeces[1]), frame_t


def data_augment(image):
    # resize and rancom crop
    image = tf.image.resize(image, [360, 360])
    # image = tf.image.resize(image, [352, 352])
    image = tf.image.random_crop(image, size=[352, 352, 9])
    # random flip
    image = tf.image.random_flip_left_right(image)
    # normalization
    # mean = tf.tile(tf.constant([0.485, 0.456, 0.406]))
    # std = tf.tile(tf.constant([0.229, 0.224, 0.225]))
    # image = (image - mean) / std
    # image = tf.image.per_image_standardization(image)
    return image


def decode_img(image: str):
    """
    Decode the image from its filename
    :param image: the image to decode
    :return: the image decoded
    """
    image = tf.io.read_file(image)
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image
