"""
    For each point cloud, generate a CSV that includes:
        keep_ratio, class_id, true_positive, true_negative, false_positive, false_negative

    Configurations
        scannet_dir: root data folder of ScanNet
        ply_dirname: folder of point cloud in PLY. [Ply_partial, Train_ply, Ply]
        K: number of nearest neighbors for label recovering of removed points
        n_scene: only effective if ply_dir == "Ply_partial". Copy n_scene point clouds only

    if ply_dirname == "Train_ply":
        point clouds in {scannet_dir}/{ply_dirname} => {scannet_dir}/{ply_dirname}_label/{K}/*.csv

    if ply_dirname == "Ply":
        point clouds in {scannet_dir}/{ply_dirname} => {scannet_dir}/{ply_dirname}_label/{K}/*.csv

    if ply_dirname == "Ply_partial":
        first n_scene point clouds in {scannet_dir}/Ply => {scannet_dir}/{ply_dirname}_label/{K}/*.csv

"""

import os
import sys
import glob
from shutil import copyfile
import time

# ------ Configuration ------
scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# Ply_partial, Train_ply, Ply
ply_dirname = "Ply_partial"

# only effective if ply_dir == "Ply_partial"
n_scene = 50

# number of nearest neighbors
K = 5

# --- end of configuration ---

ply_dir = os.path.join(scannet_dir, ply_dirname)

if ply_dirname == "Ply_partial":
    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)
    ply_files = glob.glob(os.path.join(scannet_dir, "Ply", "*.ply"))[:n_scene]
    for ply_file in ply_files:
        copyfile(ply_file, os.path.join(ply_dir, os.path.basename(ply_file)))

save_dir = os.path.join(scannet_dir, ply_dirname+"_label", str(K))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ply_files = glob.glob(os.path.join(ply_dir, "*.ply"))
index = 1
start_time = time.time()
for ply_file in ply_files:
    if (index % 10) == 0:
        print("------ {:0.0f} (s) Processing {}/{}".format(time.time() - start_time, index, len(ply_files)))

    save_file = os.path.join(save_dir, os.path.basename(ply_file)[:-4]+".csv")
    mycmd = "../Cpp/recover_full/build/recover_full {} {} {}".format(ply_file, K, save_file)
    os.system(mycmd)
    index += 1




