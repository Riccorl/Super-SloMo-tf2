import argparse
import os
import pathlib
import shutil

import cv2
import numpy as np
import tensorflow as tf

import dataset
from models.slomo_model import SloMoNet


def extract_frames(video_path: pathlib.Path, output_path: pathlib.Path):
    """
    Extract frames from videos in the input folder.
    :param video_path:
    :param output_path:
    :return:
    """
    output_filename = output_path.parent / "tmp"
    pathlib.Path(output_filename).mkdir(parents=True, exist_ok=True)
    vidcap = cv2.VideoCapture(str(video_path))

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(
            "{}/frame%04d.jpg".format(output_filename) % count, image
        )  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    return output_filename, fps, width, height


def load_dataset(data_path: pathlib.Path, n_frames: int, batch_size: int = 32):
    """
    Prepare the tf.data.Dataset for inference
    :param data_path: directory of the dataset
    :param n_frames: number of frames between frame_0 and frame_1
    :param batch_size: size of the batch
    :return: the loaded dataset
    """
    autotune = tf.data.experimental.AUTOTUNE
    ds = (
        tf.data.Dataset.list_files(str(data_path / "*"), shuffle=False)
        .window(2, 1, drop_remainder=True)
        .flat_map(lambda window: window.batch(2))
        .map(lambda x: repeat_frames(x, n_frames), num_parallel_calls=autotune)
        .flat_map(lambda *x: tf.data.Dataset.from_tensor_slices([i for i in x]))
        .map(load_frames, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )
    return ds


def repeat_frames(frames, n_frames: int):
    """
    Load the frames in the folder specified by folder_path
    :param frames: frames
    :param n_frames: number of frames between frame_0 and frame_1
    :return:
    """
    return [(frames[0], frames[1], str(f)) for f in range(1, n_frames)]


def load_frames(frames):
    """
    Load the frames in the folder specified by folder_path
    :param frames: frames
    :return: the decoded frames
    """
    frame_0 = dataset.decode_img(tf.io.read_file(frames[0]))
    frame_1 = dataset.decode_img(tf.io.read_file(frames[1]))
    return frame_0, frame_1, int(frames[2])


def deprocess(img):
    img = 255 * (img + 1.0)
    return img.numpy().astype(np.uint8)


def predict(
    video_path: pathlib.Path,
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    n_frames: int,
    fps_rate: int,
):
    batch_size = n_frames
    data_path, fps, w, h = extract_frames(video_path, output_path)

    model = SloMoNet(n_frames=n_frames)
    tf.train.Checkpoint(net=model).restore(str(model_path))
    # model.load_weights(str(model_path))
    ds = load_dataset(data_path, n_frames, batch_size)
    progbar = tf.keras.utils.Progbar(None)

    videos = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(output_path), fourcc, fps * fps_rate, (w, h))

    for step, frames in enumerate(ds):
        predictions, _ = model(frames, training=False)
        videos += (
            # [deprocess(frames[0][0])]
             [deprocess(f) for f in predictions]
            # + [deprocess(frames[0][1])]
        )
        progbar.update(step + 1)

    for f in videos:
        out_video.write(f)

    out_video.release()
    shutil.rmtree(data_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="path to input video", dest="video_path")
    parser.add_argument(help="path where to save the slomo video", dest="output_path")
    parser.add_argument("--model", help="path to model", dest="model_path")
    parser.add_argument(
        "--fps",
        help="number of fps to insert between the frames",
        dest="fps",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--slomo-factor", help="slomo factor", dest="slomo_factor", default=5, type=int,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = pathlib.Path(args.video_path)
    output_path = pathlib.Path(args.output_path)
    model_path = pathlib.Path(args.model_path)
    predict(video_path, model_path, output_path, args.fps, args.slomo_factor)


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # os.environ["OMP_NUM_THREADS"] = "12"
    main()
