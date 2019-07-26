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


git_dir = "/home/dtc/MyGit/dtc-sparseconvnet/"
val_100_pth_dir = "/home/dtc/Data/ScanNet/Pth/Original"

val_dir = os.path.join(git_dir, "val")
pth_files = glob.glob(os.path.join(val_dir, "*.pth"))
for pth_file in pth_files:
    f_src = pth_file
    f_dst = os.path.join(val_100_pth_dir, os.path.basename(pth_file))
    print(f_src + " ---> " + f_dst)
    copyfile(f_src, f_dst)

