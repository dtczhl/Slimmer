"""
    move original pth files
"""

import torch
import numpy as np
import glob
import math
import torch.utils.data
import scipy.ndimage
import multiprocessing as mp
import os
from shutil import copyfile
import sys

# ------ configuration ------

# path to this git
git_dir = "/home/dtc/MyGit/dtc-sparseconvnet/"

# path to ScanNet directory
scannet_dir = "/home/dtc/Data/ScanNet"

# --- end of configuration ---

save_dir = os.path.join(scannet_dir, "Pth/Original")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

val_dir = os.path.join(git_dir, "val")
pth_files = glob.glob(os.path.join(val_dir, "*.pth"))
for pth_file in pth_files:
    f_src = pth_file
    f_dst = os.path.join(save_dir, os.path.basename(pth_file))
    print(f_src + " ---> " + f_dst)
    copyfile(f_src, f_dst)

