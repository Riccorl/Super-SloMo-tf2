import argparse
import os
import shutil
from pathlib import Path
import random


def extract_frames(input_dir: Path, output_dir: Path, img_width: str, img_height: str):
    """
    Extract frames from videos in the input folder.
    """
    video_ext = {".m4v", ".mov", ".MOV", ".mp4"}
    for video_file in input_dir.glob("**/*"):
        if video_file.suffix in video_ext:
            output_filename = output_dir / video_file.name
            Path(output_filename).mkdir(parents=True, exist_ok=True)
            cmd = "ffmpeg -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%04d.jpg".format(
                video_file, img_width, img_height, output_filename
            )
            os.system(cmd)


def group_frames(input_dir: Path, output_dir: Path, n_frame: int = 12):
    """
    Group frames in subfolders in batch of size n_frames.
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
                        str(a), "{}/{}/{}".format(output_dir, folder_counter, a.name),
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
    train_dir = Path(args.input_dir + "train")
    tmp_dir = Path(train_dir + "/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(train_dir, tmp_dir, args.img_width, args.img_height)
    group_frames(tmp_dir, Path(args.output_dir + "train"))

    # extract test
    test_dir = Path(args.input_dir + "test")
    tmp_dir = Path(test_dir + "/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(test_dir, tmp_dir, args.img_width, args.img_height)
    group_frames(tmp_dir, Path(args.output_dir + "test"))

    # random sampling for validation
    test_files = [folder for folder in test_dir.glob("**")]
    sampled = random.sample(range(len(test_files)), 100)
    val_dir = Path(args.output_dir + "val")
    val_dir.mkdir(parents=True, exist_ok=True)
    for s in sampled:
        shutil.move("{}/{}".format(test_dir, s), "{}/{}".format(val_dir, s))
