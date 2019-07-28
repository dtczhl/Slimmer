"""
    Crop data with random sampling
"""
import torch
import glob
import os
import numpy as np
import sys

# KEEP_RATIO = 100

scannet_dir = "/home/dtc/Data/ScanNet"
data_type = "Random"

def crop_data(keep_ratio):
    print("------ Random keep ratio", keep_ratio, "------")
    original_dir = os.path.join(scannet_dir, "Pth/Original")
    dst_dir = os.path.join(scannet_dir, "Pth/{}/{}".format(data_type, keep_ratio))

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = sorted(glob.glob(os.path.join(original_dir, '*.pth')))
    for src_file in files:
        src_filename = os.path.basename(src_file)
        data = torch.load(src_file)
        coords, colors, labels = data
        new_coords, new_colors, new_labels = [], [], []
        for i in range(len(coords)):
            if np.random.rand() <= keep_ratio/100.0:
                new_coords.append(coords[i])
                new_colors.append(colors[i])
                new_labels.append(labels[i])

        new_coords = np.array(new_coords, dtype="float32")
        new_colors = np.array(new_colors, dtype="float32")
        new_labels = np.array(new_labels, dtype="float32")

        # print(new_coords[0][0].dtype)
        # print(new_colors[0][0].dtype)
        # print(new_labels[0].dtype)

        dst_file_path = os.path.join(dst_dir, src_filename)
        print(src_file, " ---> ", dst_file_path)
        new_coords = np.ascontiguousarray(new_coords)
        new_colors = np.ascontiguousarray(new_colors)
        new_labels = np.ascontiguousarray(new_labels)
        torch.save((new_coords, new_colors, new_labels), dst_file_path)


for keep_ratio in range(5, 6, 5):
    crop_data(keep_ratio)




