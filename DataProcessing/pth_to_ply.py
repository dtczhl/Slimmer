"""
    convert pth to ply
"""

import torch
import numpy as np
import glob

import os
import time
import sys
from plyfile import PlyData, PlyElement

# ------ Configuration ------

# path to Pth
Pth_dir = "/home/dtc/Backup/Data/ScanNet/Pth/Original"

# path to ply
Ply_dir = "/home/dtc/Backup/Data/ScanNet/Ply"

# --- end of Configuration ---

if not os.path.exists(Ply_dir):
    os.makedirs(Ply_dir)

pth_files = glob.glob(os.path.join(Pth_dir, "*.pth"))
for pth_file in pth_files:
    data = torch.load(pth_file)
    coords, colors, label = data
    colors = np.array((colors+1)/2 * 255, dtype="uint8")
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
    dst_ply = os.path.join(Ply_dir, os.path.basename(pth_file)[:-4]+".ply")
    plydata.write(dst_ply)
