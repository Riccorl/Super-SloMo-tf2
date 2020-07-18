import argparse
import os
import pathlib

import cv2
import numpy as np
import tensorflow as tf

import dataset
from models.slomo_model import SloMoNet


def load_dataset(data_path: pathlib.Path, batch_size: int = 32):
    """
    Prepare the tf.data.Dataset for inference
    :param data_path: directory of the dataset
    :param batch_size: size of the batch
    :return: the loaded dataset
    """
    autotune = tf.data.experimental.AUTOTUNE
    ds = (
        tf.data.Dataset.list_files(str(data_path / "*_0?.png"), shuffle=False)
        .window(2, 1, drop_remainder=True)
        .flat_map(lambda window: window.batch(2))
        .map(load_frames, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )
    return ds


def load_frames(frames):
    """
    Load the frames in the folder specified by folder_path
    :param frames: frames
    :return: the decoded frames
    """
    frame_0 = dataset.decode_img(frames[0])
    frame_1 = dataset.decode_img(frames[1])
    return frame_0, frame_1


def deprocess(image):
    """
    Convert predicted image to 255
    :param image: the image to convert
    :return: image converted
    """
    return (255 * image).numpy().astype(np.uint8)


def predict(
    data_path: pathlib.Path,
    model_path: pathlib.Path,
):
    """
    Predict the in-between frames
    :param data_path: path to data
    :param model_path: path do model checkpoint
    :return:
    """
    model = SloMoNet(n_frames=1 + 2)
    tf.train.Checkpoint(net=model).restore(str(model_path)).expect_partial()
    progbar = tf.keras.utils.Progbar(None)
    file_list = [f for f in data_path.glob("**/*_0?.png")]
    print("Number of files:", len(file_list))
    for i, j in zip(file_list[0::2], file_list[1::2]):
        out_frame_path = str(i).rsplit("_", 1)[0] + "_01_interp.png"
        frame_0, frame_1 = load_frames([str(i), str(j)])
        frames = (frame_0[None, :], frame_1[None, :])
        predictions, _ = model(frames + ([1],), training=False)
        out_frame = deprocess(predictions[0])
        cv2.imwrite(out_frame_path, cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
        progbar.add(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="path to input data", dest="data_path")
    parser.add_argument(help="path to model", dest="model_path")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = pathlib.Path(args.data_path)
    print("Data path:", data_path)
    model_path = pathlib.Path(args.model_path)
    predict(data_path, model_path)


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    main()
