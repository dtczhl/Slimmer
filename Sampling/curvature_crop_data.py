"""
    Crop data with Menger curvature sampling
"""
import torch
import glob
import os
import pandas as pd
import numpy as np
import sys

from pyntcloud import PyntCloud

# KEEP_RATIO = 100

scannet_dir = "/home/dtc/Data/ScanNet"

n_k_neighbors = 10
data_type = "Curvature"


def crop_data(keep_ratio):
    print("------ Curvature keep ratio", keep_ratio, "------")
    original_dir = os.path.join(scannet_dir, "Pth/Original")
    dst_dir = os.path.join(scannet_dir, "Pth/{}/{}".format(data_type, keep_ratio))

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = sorted(glob.glob(os.path.join(original_dir, '*.pth')))
    for src_file in files:
        src_filename = os.path.basename(src_file)
        data = torch.load(src_file)
        coords, colors, labels = data

        pd_data = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]})
        cloud = PyntCloud(pd_data)
        k_neighbors = cloud.get_neighbors(k=n_k_neighbors)
        ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        cloud.add_scalar_field("curvature", ev=ev)

        n_keep_points = int(keep_ratio / 100.0 * len(labels))

        index_sort = np.argsort(np.abs(cloud.points["curvature({})".format(n_k_neighbors+1)].to_numpy()))[::-1][:n_keep_points]
        index_sort.sort()

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




