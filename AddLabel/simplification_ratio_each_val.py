"""
    Calculate IOU for each simplified point cloud

    Run after simplification_ratio_each.py

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

add_missing_label_each_dir = os.path.join(scannet_dir, "AddMissingLabelEach")

data_type_dir = os.path.join(add_missing_label_each_dir, data_type, str(k_KNN))
if not os.path.exists(data_type_dir):
    print(data_type_dir, "does not exist")
    sys.exit(-1)


def calculate_iou_dir(keep_ratio):

    save_file = os.path.join(data_type_dir, str(keep_ratio), 'iou.csv')
    with open(save_file, "w") as f_out:

        txt_files = glob.glob(os.path.join(data_type_dir, str(keep_ratio), "*.txt"))
        for txt_file in txt_files:
            iou = 0
            n = 0
            data = np.genfromtxt(txt_file, dtype=np.int, delimiter=",", skip_header=1)
            for row_data in data:
                divisor = np.sum(row_data[1:])
                if row_data[1] + row_data[4] == 0:
                    # omit this class
                    continue
                n += 1
                if divisor != 0:
                    csi = row_data[1] / divisor
                    iou += csi
            f_out.write("{}\n".format(100*iou/n))
    print(save_file)




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


