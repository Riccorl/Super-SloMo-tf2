from __future__ import division
from __future__ import print_function
import os
import glob
import numpy as np
import cv2
from skimage.measure import compare_psnr, compare_ssim
import skimage.data
import skimage.transform
from multiprocessing import Pool
from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser("Video Interpolation Evaluation Script.")
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="directory where ground-truth intermediate frames are stored.",
    )
    parser.add_argument(
        "--gt-suffix",
        default="_gt.png",
        help="suffix of ground-truth intermediate frames",
    )
    parser.add_argument(
        "--res-dir",
        required=True,
        help="directory where interpolation results are stored",
    )
    parser.add_argument(
        "--res-suffix",
        default="_ours.png",
        help="suffix of interpolation result frames",
    )
    parser.add_argument(
        "--motion-mask-dir",
        default=None,
        help="directory where motion masks are stored (optional)",
    )
    parser.add_argument(
        "--mask-out",
        action="store_true",
        help="whether to eleminate pixels outside of motion masks for evaluation.",
    )
    parser.add_argument(
        "--buggy-motion-mask",
        action="store_true",
        help="whether the provided motion mask is buggy or not",
    )
    args = parser.parse_args()
    return args


def evaluate_single_im(
    gt_im_path,
    res_im_path,
    motion_mask_path=None,
    mask_out=False,
    buggy_motion_mask=False,
):
    ref_im = cv2.imread(gt_im_path)
    res_im = cv2.imread(res_im_path)
    assert np.all(
        ref_im.shape == res_im.shape
    ), "Dimension check failed: check interpolation results!"
    if motion_mask_path is not None:
        mask_im = cv2.imread(motion_mask_path, 0).astype(np.float32)
        # in my original implementation
        # I've converted the motion mask provided by DVF
        # to denote motion area with white (pixel value of 255)
        # so I'll do the conversion on the fly
        if buggy_motion_mask:
            mask_im = (mask_im > 128).astype(np.float32)
        else:
            mask_im = (mask_im < 128).astype(np.float32)
        assert np.all(
            ref_im.shape[:2] == mask_im.shape
        ), "Dimension check for motion mask failed."
    else:
        mask_im = np.ones((ref_im.shape[0], ref_im.shape[1]), dtype=np.float32)
        print("No motion mask")

    if not mask_out:
        # what we used in the SuperSloMo paper
        idxes = np.where(mask_im > 0.5)
        if len(idxes[0]) == 0:
            return np.array([np.nan, np.nan, np.nan])
        y1 = min(idxes[0])
        x1 = min(idxes[1])
        y2 = max(idxes[0]) + 1
        x2 = max(idxes[1]) + 1
        # [y1, x1, y2, x2] = find_roi(mask_im)
        err = res_im.astype(np.float32)[idxes] - ref_im.astype(np.float32)[idxes]
        ie = np.mean(np.sqrt(np.sum(err * err, axis=1)))
        res_im = res_im[y1:y2, x1:x2, :]
        ref_im = ref_im[y1:y2, x1:x2, :]
        if res_im.shape[0] < 10 or res_im.shape[1] < 10:
            return np.array([np.nan, np.nan, np.nan])
    else:
        # what DVF used in their paper
        # a little bit over estimated
        idxes = np.where(mask_im <= 0.5)
        mask_im *= 0
        mask_im[idxes] = 1
        # idxes = np.where(mask_im > 0.5)
        # mask_im[idxes] = 0
        res_im = res_im.astype(np.float32) * np.tile(
            mask_im[:, :, np.newaxis], (1, 1, 3)
        )
        ref_im = ref_im.astype(np.float32) * np.tile(
            mask_im[:, :, np.newaxis], (1, 1, 3)
        )
        err = res_im - ref_im
        ie = np.mean(np.sqrt(np.sum(err * err, axis=2)))

    res_im = res_im.astype(np.uint8)
    ref_im = ref_im.astype(np.uint8)
    psnr = compare_psnr(res_im, ref_im)
    ssim = compare_ssim(res_im, ref_im, multichannel=True, gaussian_weights=True)
    return np.array([psnr, ssim, ie])


def process_single_seq(gt_seq_dir, args):
    gt_suffix = args.gt_suffix
    res_dir = args.res_dir
    res_suffix = args.res_suffix
    motion_mask_dir = args.motion_mask_dir
    mask_out = args.mask_out
    buggy_motion_mask = args.buggy_motion_mask

    gt_im_paths = glob.glob(os.path.join(gt_seq_dir, "*" + gt_suffix))
    assert len(gt_im_paths) > 0, "No ground-truth data was found."

    avg_metrics = np.zeros((0, 3), dtype=float)
    for gt_im_path in gt_im_paths:
        seq_dir, im_name = os.path.split(gt_im_path)
        _, seq_name = os.path.split(seq_dir)
        res_im_path = os.path.join(
            res_dir, seq_name, im_name[: -len(gt_suffix)] + res_suffix
        )
        assert os.path.exists(
            res_im_path
        ), "Interpolation result {} was not found.".format(res_im_path)
        if motion_mask_dir is not None:
            mask_im_path = os.path.join(
                motion_mask_dir,
                seq_name,
                "motion_mask.png",  # problematic, but let's use DVF's naming fashion
            )
        else:
            mask_im_path = None
        m = evaluate_single_im(
            gt_im_path, res_im_path, mask_im_path, mask_out, buggy_motion_mask
        )
        avg_metrics = np.vstack((avg_metrics, m))
    return avg_metrics


def main(args):
    gt_dir = args.gt_dir
    seqs = glob.glob(os.path.join(gt_dir, "*"))
    # print('There are {} sequences for evaluation.'.format(len(seqs)))

    # PSNR, SSIM, IE
    avg_metrics = np.zeros((0, 3), dtype=float)
    for s in tqdm(seqs):
        metrics = process_single_seq(s, args)
        avg_metrics = np.vstack((avg_metrics, metrics))
    avg_metrics = np.nanmean(avg_metrics, axis=0)
    print(
        "PSNR: {:.2f}, SSIM: {:.3f}, IE: {:.2f}".format(
            avg_metrics[0], avg_metrics[1], avg_metrics[2]
        )
    )

    # ref_im_path = '/home/hzjiang/Downloads/ucf101_interp_ours/1/frame_01_gt.png'
    # res_im_path = '/home/hzjiang/Downloads/ucf101_interp_ours/1/frame_01_ours.png'
    # print(evaluate_single_im(ref_im_path, res_im_path))


if __name__ == "__main__":
    args = parse_args()

    main(args)
