"""
    generate labels for training a predictor for simplification ratio
"""

import os
import sys
import glob
from shutil import copyfile


# ------ Configuration ------
scannet_dir = "/home/dtc/Backup/Data/ScanNet"


# Ply_partial, Train_ply
ply_dirname = "Train_ply"

# only effective if ply_dir == "Ply_partial"
n_scene = 50

K = 1

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
for ply_file in ply_files:
    if (index % 10) == 0:
        print("------ Processing {}/{}".format(index, len(ply_files)))

    save_file = os.path.join(save_dir, os.path.basename(ply_file)[:-4]+".csv")
    mycmd = "../Cpp/recover_full/build/recover_full {} {} {}".format(ply_file, K, save_file)
    os.system(mycmd)
    index += 1


# if ply_dirname == "Ply_partial":
#     ply_dir = os.path.join(scannet_dir, "Ply_partial")
#     if not os.path.exists(ply_dir):
#         os.makedirs(ply_dir)
#     ply_files = glob.glob(os.path.join(scannet_dir, "Ply", "*.ply"))[:n_scene]
#     for ply_file in ply_files:
#         copyfile(ply_file, os.path.join(ply_dir, os.path.basename(ply_file)))
#
#     save_dir = os.path.join(scannet_dir, "Ply_partial_label", str(K))
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     ply_files = glob.glob(os.path.join(ply_dir, "*.ply"))
#     index = 1
#     for ply_file in ply_files:
#         if (index % 10) == 0:
#             print("------ Processing {}/{}".format(index, len(ply_files)))
#
#         save_file = os.path.join(save_dir, os.path.basename(ply_file)[:-4]+".csv")
#         mycmd = "../Cpp/recover_full/build/recover_full {} {} {}".format(ply_file, K, save_file)
#         os.system(mycmd)
#         index += 1



