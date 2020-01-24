"""
    transform pth to ply under {scannet_dir}/Train_ply dir for training simplification ratio predictor
"""


import torch
import numpy as np
import glob
import os
import sys
from plyfile import PlyData, PlyElement
import time

# ------ Configuration ------
scannet_dir = "/home/dtc/Backup/Data/ScanNet"
# --- end of Configuration ---

# train -> {scannet_dir}/Train_ply
pth_files = glob.glob(os.path.join("../train", "*.pth"))
ply_dir = os.path.join(scannet_dir, "Train_ply")
if not os.path.exists(ply_dir):
    os.makedirs(ply_dir)

index = 1
for pth_file in pth_files:

    time_start = time.time()

    data = torch.load(pth_file)
    coords, colors, label = data
    colors = np.array((colors + 1) / 2 * 255, dtype="uint8")
    ply_save = []
    for i in range(len(coords)):
        ply_save.append((coords[i][0], coords[i][1], coords[i][2],
                         colors[i][0], colors[i][1], colors[i][2], label[i]))
    ply_save = np.array(ply_save,
                        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                               ("label", "u1")])
    el = PlyElement.describe(ply_save, "vertex")
    plydata = PlyData([el], text=True)
    dst_ply = os.path.join(ply_dir, os.path.basename(pth_file)[:-4]+".ply")
    plydata.write(dst_ply)

    if index % 10 == 0:
        print("{}: {}/{}".format(time.time() - time_start, index, len(pth_files)))
    index += 1




