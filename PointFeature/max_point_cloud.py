"""
    The point cloud that has the largest number of points
"""

import numpy as np
import torch
import glob
import os
import sys

scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# path to pth
original_dir = os.path.join(scannet_dir, "Pth/Original")

pth_files = glob.glob(os.path.join(original_dir, "*.pth"))

n_points_max = 0
name_points_max = ''

for pth_file in pth_files:
    data = torch.load(pth_file)
    coords, colors, labels = data

    n_points = len(coords)
    if n_points > n_points_max:
        n_points_max = n_points
        name_points_max = os.path.basename(pth_file)


print(name_points_max)
print(n_points_max)
