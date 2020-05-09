"""
    add label to missing points

    From PlyLabel -> AddMissingLabel
"""

import numpy as np
import glob
import os
import pandas as pd
import sys
import datetime
import shutil

# ------ Configuration ------

scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# Random, Grid, Hierarchy
data_type = "Hierarchy"

device = "alienware"

specify_id = []  # if want to valid specific ids

k_KNN = 5  # number of nearest labels

# --- end of configuration ---

time_save_dir = os.path.join("../Result/KnnTime", device, data_type, str(k_KNN))
if not os.path.exists(time_save_dir):
    os.makedirs(time_save_dir)

tmp_dir = "../tmp/"
# clear tmp
files = glob.glob(os.path.join(tmp_dir, "*"))
for file in files:
    os.remove(file)

ply_label_dir = "PlyLabel"
add_label_dir = "AddMissingLabel"

original_ply_dir = os.path.join(scannet_dir, "Ply")
if not os.path.exists(original_ply_dir):
    print("{} does not exist! See DataProcessing/pth_to_ply.py".format(original_ply_dir))
    sys.exit(1)


def add_label_KNN(keep_ratio, k_KNN):

    simplified_pth_dir = os.path.join(scannet_dir, ply_label_dir, data_type, str(keep_ratio))
    if not os.path.exists(simplified_pth_dir):
        print("{} does not exist! See Validation/memory_valid.py".format(simplified_pth_dir))
        sys.exit(1)

    save_dir = os.path.join(scannet_dir, add_label_dir, data_type, str(k_KNN), str(keep_ratio))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pred_files = glob.glob(os.path.join(simplified_pth_dir, "*.ply"))
    for pred_file in pred_files:
        pred_base_name = os.path.basename(pred_file)

        orig_file = os.path.join(scannet_dir, "Ply", pred_base_name)
        save_file = os.path.join(save_dir, pred_base_name[:-4]+".txt")

        mycmd = "../Cpp/add_label/build/add_label {} {} {} {}"\
                .format(orig_file, pred_file, save_file, k_KNN)
        os.system(mycmd)

    time_save_file = os.path.join(time_save_dir, "time.txt.{}".format(keep_ratio))
    print("saving file to:", time_save_file)
    shutil.move(os.path.join(tmp_dir, "time.txt"), time_save_file)
    # clear tmp
    tmp_files = glob.glob(os.path.join(tmp_dir, "*"))
    for tmp_file in tmp_files:
        os.remove(tmp_file)

    print("{} --- {} k: {} keep ratio: {}".format(str(datetime.datetime.now()), data_type, str(k_KNN), str(keep_ratio)))


if __name__ == "__main__":

    def func_filename(x):
        return int(os.path.basename(x))

    if specify_id:
        for my_id in specify_id:
            add_label_KNN(my_id, k_KNN)
    else:
        data_dirs = sorted(glob.glob(os.path.join(scannet_dir, ply_label_dir, data_type, "*")), key=func_filename)
        for data_dir in data_dirs:
            my_id = int(os.path.basename(data_dir))
            add_label_KNN(my_id, k_KNN)


