"""
    Pth dir to PlyDirs
"""

import torch
import numpy as np
import glob

import os
import time
import sys
from plyfile import PlyData, PlyElement


# ------ Configuration ------

scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# Random, Hierarchy, Grid
data_type = "Random"

n_scene = 30

# --- end of Configuration ---

def func_filename(x):
    return int(os.path.basename(x))


def pth_to_ply(id):
    print("------ Processing {}".format(id))
    dirname = os.path.join(scannet_dir, "Pth", data_type, str(id))
    pth_files = glob.glob(os.path.join(dirname, "*.pth"))[:n_scene]
    for pth_file in pth_files:
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
        Ply_dir = os.path.join(scannet_dir, "PlyDirs", data_type, str(id))
        if not os.path.exists(Ply_dir):
            os.makedirs(Ply_dir)
        dst_ply = os.path.join(Ply_dir, os.path.basename(pth_file)[:-4] + ".ply")
        plydata.write(dst_ply)


data_dirs = sorted(glob.glob(os.path.join(scannet_dir, "Pth", data_type, "*")), key=func_filename)

for data_dir in data_dirs:
    my_id = int(os.path.basename(data_dir))
    pth_to_ply(my_id)

