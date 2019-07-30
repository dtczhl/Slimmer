"""
    Crop data with Menger curvature sampling
"""
import torch
import glob
import os
import pandas as pd
import numpy as np
import sys
from shutil import copyfile

from pyntcloud import PyntCloud

# KEEP_RATIO = 100

scannet_dir = "/home/dtc/Data/ScanNet"

n_k_neighbors = 10
data_type = "Grid"


def crop_data(keep_ratio_overall):
    print("------ {} keep ratio {} ------".format(data_type, keep_ratio_overall))

    original_dir = os.path.join(scannet_dir, "Pth/Original")
    dst_dir = os.path.join(scannet_dir, "Pth/{}".format(data_type))

    for keep_ratio in range(keep_ratio_overall[0], keep_ratio_overall[1]+1, keep_ratio_overall[2]):
        if not os.path.exists(os.path.join(dst_dir, "{}".format(keep_ratio))):
            os.makedirs(os.path.join(dst_dir, "{}".format(keep_ratio)))

    files = sorted(glob.glob(os.path.join(original_dir, '*.pth')))
    for src_file in files:
        src_filename = os.path.basename(src_file)
        data = torch.load(src_file)
        coords, colors, labels = data
        coords = np.array(coords, "float32")
        colors = np.array(colors, "float32")
        labels = np.array(labels, "float32")

        # copy file
        original_data = np.c_[coords, colors, labels]
        tmp_dir = "../tmp/"
        tmp_file_name = os.path.join(tmp_dir, src_filename)
        original_data.ravel().tofile(tmp_file_name)

        mycmd = "../Cpp/sample_data/build/sample_data {} {} {} {} {} {}"\
            .format(data_type.lower(), keep_ratio_overall[0], keep_ratio_overall[1], keep_ratio_overall[2], dst_dir, src_filename)
        os.system(mycmd)
        os.remove(tmp_file_name)

        for keep_ratio in range(keep_ratio_overall[0], keep_ratio_overall[1]+1, keep_ratio_overall[2]):
            src_trim_file = tmp_file_name + ".{}".format(keep_ratio)
            if not os.path.exists(src_trim_file):
                sys.exit("Error, file " + src_trim_file + " does not exist")

            # print(src_trim_file)
            new_data = np.fromfile(src_trim_file, "<f4")
            new_data = np.reshape(new_data, (-1, 7))
            os.remove(src_trim_file)

            new_coords = new_data[:, 3]
            new_colors = new_data[:, 3:6]
            new_labels = new_data[:, 6]

            new_coords = np.array(new_coords, "float32")
            new_colors = np.array(new_colors, "float32")
            new_labels = np.array(new_labels, "float32")

            dst_file_path = os.path.join(dst_dir, "{}/{}".format(keep_ratio, src_filename))
            print(src_file, " ---> ", dst_file_path)
            new_coords = np.ascontiguousarray(new_coords)
            new_colors = np.ascontiguousarray(new_colors)
            new_labels = np.ascontiguousarray(new_labels)
            torch.save((new_coords, new_colors, new_labels), dst_file_path)


# start, end(include), step
crop_data([10, 90, 10])




