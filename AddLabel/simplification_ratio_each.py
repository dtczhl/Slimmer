"""
    Log TP FP FN for each simplified point cloud

    AddMissingLabel -> AddMissingLabelEach

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

specify_id = []  # if want to valid specific ids

k_KNN = 1  # number of nearest labels

# --- end of configuration ---


save_main_dir = os.path.join(scannet_dir, "AddMissingLabelEach", data_type, str(k_KNN))
if not os.path.exists(save_main_dir):
    os.makedirs(save_main_dir)

add_missing_label_dir = os.path.join(scannet_dir, "AddMissingLabel")

data_type_dir = os.path.join(add_missing_label_dir, data_type, str(k_KNN))
if not os.path.exists(data_type_dir):
    print(data_type_dir, "does not exist")
    sys.exit(-1)


def calculate_iou_dir(keep_ratio):

    save_dir = os.path.join(save_main_dir, str(keep_ratio))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    txt_files = glob.glob(os.path.join(data_type_dir, str(keep_ratio), "*.txt"))
    for txt_file in txt_files:
        save_file = os.path.join(save_dir, os.path.basename(txt_file))
        with open(save_file, "w") as f_out:
            f_out.write("ClassID,TruePositive,FalsePositive,TrueNegative,FalseNegative\n")
            confusion_matrice = np.zeros((20, 4), dtype=np.int)
            data = np.loadtxt(txt_file, usecols=(6, 7), dtype=np.int)
            valid_idx = (data[:, 0] >= 0) & (data[:, 0] <= 19)
            data = data[valid_idx, :]
            for row_data in data:
                if row_data[0] == row_data[1]:
                    # TP
                    confusion_matrice[row_data[0]][0] += 1
                else:
                    # FN
                    confusion_matrice[row_data[0]][3] += 1
                    # FP
                    confusion_matrice[row_data[1]][1] += 1

            for i_row in range(len(confusion_matrice)):
                f_out.write("{},{},{},{},{}\n".format(i_row, confusion_matrice[i_row][0], confusion_matrice[i_row][1],
                                                      confusion_matrice[i_row][2], confusion_matrice[i_row][3]))

    print(datetime.datetime.now(), os.path.join(data_type_dir, str(keep_ratio)), '->', save_dir)


if __name__ == "__main__":
    def func_filename(x):
        return int(os.path.basename(x))

    if specify_id:
        for my_id in specify_id:
            calculate_iou_dir(my_id)
    else:
        data_dirs = sorted(glob.glob(os.path.join(data_type_dir, "*")), key=func_filename)
        for data_dir in data_dirs:
            my_id = int(os.path.basename(data_dir))
            calculate_iou_dir(my_id)


