"""
    generate labels for training a predictor for simplification ratio
"""

import os
import sys
import glob
from shutil import copyfile


scannet_dir = "/home/dtc/Backup/Data/ScanNet"

ply_dir = os.path.join(scannet_dir, "Ply_partial")

n_scene = 30


Ply_partial_dir = os.path.join(scannet_dir, "Ply_partial")
if not os.path.exists(Ply_partial_dir):
    os.makedirs(Ply_partial_dir)
    ply_files = glob.glob(os.path.join(scannet_dir, "Ply", "*.ply"))[:n_scene]
    for ply_file in ply_files:
        copyfile(ply_file, os.path.join(Ply_partial_dir, os.path.basename(ply_file)))


mycmd = "../Cpp/recover_full/build/recover_full {}".format(ply_dir)
os.system(mycmd)


