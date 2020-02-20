import argparse
import os
import random
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(input_dir: Path, output_dir: Path, width: str, height: str):
    """
    Extract frames from videos in the input folder.
    :param input_dir:
    :param output_dir:
    :param width:
    :param height:
    :return:
    """
    video_ext = {".m4v", ".mov", ".MOV", ".mp4"}
    for video_file in tqdm(input_dir.glob("**/*")):
        if video_file.suffix in video_ext:
            output_filename = output_dir / video_file.name
            Path(output_filename).mkdir(parents=True, exist_ok=True)
            # vidcap = cv2.VideoCapture(str(video_file))
            # success, image = vidcap.read()
            # count = 0
            # while success:
            #     image = cv2.resize(image, (width, height))
            #     cv2.imwrite("{}/frame%04d.jpg".format(output_filename) % count, image)  # save frame as JPEG file
            #     success, image = vidcap.read()
            #     count += 1
            cmd = "ffmpeg -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%04d.jpg".format(
                video_file, width, height, output_filename
            )
            os.system(cmd)


def group_frames(input_dir: Path, output_dir: Path, n_frame: int = 12):
    """
    Group frames in subfolders in batch of size n_frames.
    :param input_dir:
    :param output_dir:
    :param n_frame:
    :return:
    """
    folder_counter = 0
    for folder in input_dir.glob("**"):
        files = sorted(f for f in folder.glob("*") if f.is_file())
        accumulator = []
        for file in files:
            accumulator.append(file)
            if len(accumulator) >= n_frame:
                Path("{}/{}".format(output_dir, folder_counter)).mkdir(
                    parents=True, exist_ok=True
                )
                for a in accumulator:
                    shutil.move(
                        str(a), "{}/{}/{}".format(output_dir, folder_counter, a.name)
                    )
                accumulator = []
                folder_counter += 1


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="path to the input folder"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="path to the output folder"
    )
    parser.add_argument("--img_width", type=int, default=640, help="output image width")
    parser.add_argument(
        "--img_height", type=int, default=360, help="output image height"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()

    # extract train
    train_dir = Path(args.input_dir) / "train"
    train_dir_out = Path(args.output_dir) / "train"
    tmp_dir = train_dir_out / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(train_dir, tmp_dir, args.img_width, args.img_height)
    group_frames(tmp_dir, train_dir_out)
    shutil.rmtree(tmp_dir)

    # extract test
    test_dir = Path(args.input_dir) / "test"
    test_dir_out = Path(args.output_dir) / "test"
    tmp_dir = test_dir_out / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(test_dir, tmp_dir, args.img_width, args.img_height)
    group_frames(tmp_dir, test_dir_out)
    shutil.rmtree(tmp_dir)

    # random sampling for validation
    test_files = [folder for folder in test_dir_out.glob("**")]
    sampled = random.sample(range(len(test_files)), 100)
    val_dir = Path(args.output_dir) / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    for s in sampled:
        shutil.move("{}/{}".format(test_dir_out, s), "{}/{}".format(val_dir, s))
