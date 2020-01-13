"""
    convert pth to formatted binary data
    x, y, z quantification
"""

import torch
import numpy as np
import glob
import math
import os
import time
import sys
from plyfile import PlyData, PlyElement

# ------ Configuration ------

scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# Original, Random, Grid, Hierarchy
data_type = "Random"

# reduce number of scenes
n_scene = 30

# --- end of Configuration ---

dimension = 3
scale = 100
full_scale = 4096

# path to Pth
Pth_dir = os.path.join(scannet_dir, "Pth", data_type)

# path to Bin
Bin_dir = os.path.join(scannet_dir, "Bin")
if not os.path.exists(Bin_dir):
    os.makedirs(Bin_dir)


def convert_pth_to_bin(save_dir, f_pth):
    def coords_transform(physical_val):
        a, b, c = physical_val
        m = np.eye(3)
        m *= scale
        # theta = np.random.rand()*2*math.pi
        theta = 0
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m) + full_scale / 2
        m = a.min(0)
        M = a.max(0)
        q = M - m
        offset = -m
        a += offset
        idxs = (a.min(1) >= 0) * (a.max(1) < full_scale)
        a = a[idxs]
        b = b[idxs]
        c = c[idxs]
        return a, b, c

    data = torch.load(f_pth)
    coords, colors, labels = coords_transform(data)
    coords = torch.from_numpy(coords).long()
    coords = torch.cat([coords, torch.LongTensor(coords.shape[0], 1).fill_(0)], 1)
    colors = torch.from_numpy(colors)
    original_data = np.c_[np.array(coords.detach().numpy()[:, :3], "float32"),
                          np.array(colors.detach().numpy(), "float32"),
                          np.array(labels, "float32")]
    save_data_file = os.path.join(save_dir, os.path.basename(f_pth)[:-4] + ".bin")
    original_data.ravel().tofile(save_data_file)


if data_type.lower() == "original":
    save_dir = os.path.join(Bin_dir, "Original")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pth_files = glob.glob(os.path.join(Pth_dir, "*.pth"))[:n_scene]
    for i in range(len(pth_files)):
        if (i+1) % 10 == 0:
            print("--- {} {}/{}".format(data_type, i+1, len(pth_files)))
        convert_pth_to_bin(save_dir, pth_files[i])
    print("Done")
else:
    def func_filename(x):
        return int(os.path.basename(x))
    data_dirs = sorted(glob.glob(os.path.join(scannet_dir, "Pth", data_type, "*")), key=func_filename)
    for data_dir in data_dirs:
        print("--- {} {}".format(data_type, os.path.basename(data_dir)))
        save_dir = os.path.join(Bin_dir, data_type, os.path.basename(data_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pth_files = glob.glob(os.path.join(data_dir, "*.pth"))[:n_scene]
        for pth_file in pth_files:
            convert_pth_to_bin(save_dir, pth_file)

