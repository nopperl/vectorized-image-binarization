#!/usr/bin/env python3
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import join, splitext

import cv2
from collections import OrderedDict
import numpy as np
import skimage
import sklearn.metrics



def binarize(img, w=3, n_min=3, eps=1e-10, divisor="N", mean_divisor="N_e"):
    if type(img) is str:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = img.astype(float)
    
    # construct contrast image
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(img, w // 2, mode="edge"), (w,w))
    local_max = windows.max(axis=(2,3))
    local_min = windows.min(axis=(2,3))
    contrast = (local_max - local_min) / (local_max + local_min + eps)
    
    # find high-contrast pixels
    threshold, hi_contrast = cv2.threshold((contrast * 255).astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    del contrast
    hi_contrast = hi_contrast.astype(float)  # Su et al. also invert
    hi_contrast /= 255
    hi_contrast_windows = np.lib.stride_tricks.sliding_window_view(np.pad(hi_contrast, w // 2, mode="edge"), (w,w))

    # classify pixels
    hi_contrast_count = hi_contrast_windows.sum(axis=(2,3))
    
    if mean_divisor == "N":
        e_mean = np.mean(windows * hi_contrast_windows, axis=(2,3))  # matrix multiplication in axes 2 and 3
    else:
        e_sum = np.sum(windows * hi_contrast_windows, axis=(2,3))  # matrix multiplication in axes 2 and 3
        if mean_divisor == "N_e":
            e_mean = e_sum / hi_contrast_count  # produces nan if hi_contrast_count == 0, but since only pixels with hi_contrast_count >= n_min are considered, these values are ignored anyway
        elif mean_divisor == "2":
            e_mean = e_sum / 2
        else:
            raise ValueError
    e_mean = np.where(np.isnan(e_mean), 0, e_mean)
    if divisor == "N":
        e_std = np.square((windows - np.expand_dims(e_mean, axis=(2,3))) * hi_contrast_windows).mean(axis=(2,3))
    else:
        e_std = np.square((windows - np.expand_dims(e_mean, axis=(2,3))) * hi_contrast_windows).sum(axis=(2,3))
        if divisor == "2":
            e_std /= 2
        elif divisor == "N_e":
            e_std /= hi_contrast_count
        else:
            raise ValueError
    del windows, hi_contrast_windows
    e_std = np.sqrt(e_std)
    e_std = np.where(np.isnan(e_std), 0, e_std)
    result = np.zeros_like(img)
    result[(hi_contrast_count >= n_min) & (img <= e_mean + e_std / 2)] = 1

    return result

def evaluate(img_dir="dibco2009/DIBC02009_Test_images-handwritten", w=3, n_min=3, eps=1e-10, divisor="N", mean_divisor="N_e"):
    gt_paths = glob(join(img_dir, "*_gt.tif"))
    input_paths = [path.replace("_gt.tif", ".tif") for path in gt_paths]
    f1_scores = []
    psnrs = []
    
    for gt_path, input_path in zip(gt_paths, input_paths):
        result = binarize(input_path, w=w, n_min=n_min, eps=eps, divisor=divisor, mean_divisor=mean_divisor)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = 1 - (gt / 255)
        f1_scores.append(sklearn.metrics.f1_score(gt.ravel(), result.ravel()))
        psnrs.append(skimage.metrics.peak_signal_noise_ratio(gt, result))
    return {"f1": np.mean(f1_scores), "psnr": np.mean(psnrs)}


def split_image_channels(img_file):
    image = cv2.imread(img_file)
    return cv2.split(image)


def run_split(args):
    img_file = args.img
    (R, G, B) = split_image_channels(img_file)
    file_path, file_ex = splitext(img_file)
    cv2.imwrite(f"{file_path}-R{file_ex}", R)
    cv2.imwrite(f"{file_path}-G{file_ex}", G)
    cv2.imwrite(f"{file_path}-B{file_ex}", B)


def run_binarize(args):
    img_file = args.img
    file_path, file_ex = splitext(img_file)
    result = binarize(img_file, w=args.w, n_min=args.n_min, eps=args.eps, divisor=args.divisor, mean_divisor=args.mean_divisor)
    cv2.imwrite(f"{file_path}-binarized{file_ex}", (1 - result.astype("uint8")) * 255)

def run_evaluate(args):
    result = evaluate(args.img, w=args.w, n_min=args.n_min, eps=args.eps, divisor=args.divisor, mean_divisor=args.mean_divisor)
    print(result)

def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_split = subparsers.add_parser("split")
    parser_split.add_argument("img", metavar="FILE")
    parser_split.set_defaults(func=run_split)
    parser_classify = subparsers.add_parser("binarize")
    parser_classify.add_argument("img", metavar="FILE")
    parser_classify.add_argument("-w", default=3, type=int)
    parser_classify.add_argument("--n_min", default=3, type=int)
    parser_classify.add_argument("--eps", default=1e-10, type=float)
    parser_classify.add_argument("--divisor", default="N")
    parser_classify.add_argument("--mean_divisor", default="N_e")
    parser_classify.set_defaults(func=run_binarize)
    parser_classify = subparsers.add_parser("evaluate")
    parser_classify.add_argument("img", metavar="DIR")
    parser_classify.add_argument("-w", default=3, type=int)
    parser_classify.add_argument("--n_min", default=3, type=int)
    parser_classify.add_argument("--eps", default=1e-10, type=float)
    parser_classify.add_argument("--divisor", default="N")
    parser_classify.add_argument("--mean_divisor", default="N_e")
    parser_classify.set_defaults(func=run_evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
