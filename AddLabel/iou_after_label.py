"""
    Calculate IOU with added labels
"""

import numpy as np
import glob
import os
import iou
import sys
import datetime

# ------ Configurations ------

scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# Random, Grid, Hierarch
data_type = "Random"

# --- for saving...
device = "alienware"
model_name = "scannet_m16_rep2_residualTrue-000000650.pth"

k_KNN = 1  # number of nearest labels

specify_id = []  # if want to valid specific ids

# --- end of Configurations ---

save_dir = os.path.join("../Result", device, os.path.splitext(model_name)[0], data_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def calculate_iou_with_added_label(keep_ratio, k_KNN):

    orig_pred_txt_dir = os.path.join(scannet_dir, "AddMissingLabel", data_type, str(k_KNN), str(keep_ratio))
    if not os.path.exists(orig_pred_txt_dir):
        print("{} does not exist!".format(orig_pred_txt_dir))
        sys.exit(-1)

    txt_files = glob.glob(os.path.join(orig_pred_txt_dir, "*.txt"))

    gt = []
    pred = []
    for txt_file in txt_files:
        data = np.loadtxt(txt_file)
        valid_index = (data[:, 6] >= 0) & (data[:, 6] <= 19)
        data = data[valid_index, :]
        for orig_label in data[:, 6]:
            gt.append(orig_label)
        for pred_label in data[:, 7]:
            pred.append(pred_label)

    gt = np.array(gt, dtype="int64")
    pred = np.array(pred, dtype="int64")

    iou_ret = iou.evaluate(pred, gt)

    print("{} --- {} k: {} keep ratio: {}".format(str(datetime.datetime.now()), data_type, str(k_KNN), str(keep_ratio)))
    return keep_ratio, k_KNN, 100*iou_ret


if __name__ == "__main__":
    result = []

    def func_filename(x):
        return int(os.path.basename(x))

    if specify_id:
        for my_id in specify_id:
            result.append(calculate_iou_with_added_label(my_id, k_KNN))
    else:
        data_dirs = sorted(glob.glob(os.path.join(scannet_dir, "Pth", data_type, "*")), key=func_filename)
        for data_dir in data_dirs:
            my_id = int(os.path.basename(data_dir))
            result.append(calculate_iou_with_added_label(my_id, k_KNN))

    result_vstack = np.vstack(result)
    print("id, iou(%)")
    print(np.array_str(result_vstack, precision=2, suppress_small=True))

    save_file = os.path.join(save_dir, "iou_knn_{}".format(str(k_KNN)) + ".csv")
    print("saving file to:", save_file)
    np.savetxt(save_file, result, fmt="%d,%d,%.2f",
               header="data_id,k_KNN,IOU(%)")
