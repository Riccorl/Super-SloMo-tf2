import argparse
import os
import shutil
from pathlib import Path


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ffmpeg_dir", type=str, required=True, help="path to ffmpeg.exe"
    )
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


def extract_frames(
    input_dir: Path, output_dir: Path, ffmpeg_cmd: str, img_width: str, img_height: str
):
    """
    Extract frames from videos in the input folder.
    """
    video_ext = {".m4v", ".mov", ".MOV", ".mp4"}
    for video_file in input_dir.glob("**/*"):
        if video_file.suffix in video_ext:
            output_filename = output_dir / video_file.name
            cmd = "{} -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%04d.jpg".format(
                ffmpeg_cmd, video_file, img_width, img_height, output_filename
            )
            os.system(cmd)


def group_frames(input_dir: Path, output_dir: Path, n_frame: int = 12):
    """
    Group frames in subfolders in batch of size n_frames.
    """
    folder_counter = 0
    for folder in input_dir.glob("**"):
        files = [f for f in folder.glob("*")]
        for i in range(len(files), step=n_frame):
            os.mkdir("{}/{}".format(output_dir, folder_counter))
            for j in range(n_frame):
                shutil.move(
                    "{}/{}/{}".format(input_dir, folder.name, files[i - j].name),
                    "{}/{}/{}".format(output_dir, folder_counter, files[i - j].name),
                )
            folder_counter += 1


if __name__ == "__main__":
    args = args_parser()
    ffmpeg_cmd = os.path.join(args.ffmpeg_dir, "ffmpeg")
    tmp_dir = Path(args.output_dir + "/tmp")
    extract_frames(
        Path(args.input_dir), tmp_dir, ffmpeg_cmd, args.img_width, args.img_height
    )
    group_frames(tmp_dir, Path(args.output_dir))
