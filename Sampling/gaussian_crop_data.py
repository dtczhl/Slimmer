"""
    Bayesian Gaussian Mixture Model
"""

import torch
import glob
import os
import pandas as pd
import numpy as np
import sys

from sklearn.mixture import BayesianGaussianMixture

scannet_dir = "/home/dtc/Data/ScanNet"

data_type = "Gaussian"


def crop_data(keep_ratio):
    print("------ {} keep ratio {} ------".format(data_type, keep_ratio))
    original_dir = os.path.join(scannet_dir, "Pth/Original")
    dst_dir = os.path.join(scannet_dir, "Pth/{}/{}".format(data_type, keep_ratio))

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = sorted(glob.glob(os.path.join(original_dir, '*.pth')))
    for src_file in files:
        src_filename = os.path.basename(src_file)
        data = torch.load(src_file)
        coords, colors, labels = data

        bgm = BayesianGaussianMixture(n_components=30, n_init=2)
        bgm.fit(coords)
        print(bgm.weights_)
        sys.exit(0)

        new_coords = np.array(coords[index_sort], dtype="float32")
        new_colors = np.array(colors[index_sort], dtype="float32")
        new_labels = np.array(labels[index_sort], dtype="float32")

        dst_file_path = os.path.join(dst_dir, src_filename)
        print(src_file, " ---> ", dst_file_path)
        new_coords = np.ascontiguousarray(new_coords)
        new_colors = np.ascontiguousarray(new_colors)
        new_labels = np.ascontiguousarray(new_labels)
        torch.save((new_coords, new_colors, new_labels), dst_file_path)


for keep_ratio in range(10, 101, 10):
    crop_data(keep_ratio)




