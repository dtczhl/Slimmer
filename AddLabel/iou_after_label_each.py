"""
    Calculate IOU with added labels for each point cloud in training dataset
"""

import numpy as np
import glob
import os
import iou
import sys
import datetime
import csv

# ------ Configurations ------

scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# --- for saving...
device = "alienware"
model_name = "scannet_m32_rep2_residualTrue-000000670.pth"

k_KNN = 1  # number of nearest labels

# --- end of Configurations ---
nRatio = 100
nClass = 20

data_type = "Random"
save_dir = os.path.join("../Result", device, os.path.splitext(model_name)[0], data_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir, "train_label_gt.csv")


def iou_for_point_cloud(csv_file):
    ret = np.zeros(shape=(1, nRatio))
    data = np.zeros(shape=(nRatio, nClass, 4))
    with open(csv_file, "r") as f:
        csv_reader = csv.reader(f)
        fields = next(csv_reader)
        for row in csv_reader:
            data[int(row[0])-1, int(row[1]), 0] = int(row[2])
            data[int(row[0])-1, int(row[1]), 1] = int(row[3])
            data[int(row[0])-1, int(row[1]), 2] = int(row[4])
            data[int(row[0])-1, int(row[1]), 3] = int(row[5])

    for i_ratio in range(nRatio):
        ratio = i_ratio + 1
        CSI = np.zeros(shape=(nClass))
        n_effective_class = 0
        for i_class in range(nClass):
            TP = data[i_ratio, i_class, 0]
            FP = data[i_ratio, i_class, 2]
            FN = data[i_ratio, i_class, 3]
            denominator = TP + FP + FN
            if denominator > 0:
                n_effective_class += 1
                CSI[i_class] = TP / denominator

        ret[0, i_ratio] = np.sum(CSI) / n_effective_class
    return ret


csv_files = glob.glob(os.path.join(scannet_dir, "Train_ply_label", str(k_KNN), "*.csv"))
iou_data = np.zeros(shape=(len(csv_files), 100))
for i in range(len(csv_files)):
    iou_data[i, :] = iou_for_point_cloud(csv_files[i])

print("saving file to:", save_file)
with open(save_file, "w") as f_out:
    f_out.write("filename")
    for i_ratio in range(nRatio):
        f_out.write(",{}%".format(i_ratio+1))
    f_out.write("\n")
    for i_row in range(iou_data.shape[0]):
        f_out.write(os.path.basename(csv_files[i_row])[:-4])
        for i_col in range(iou_data.shape[1]):
            f_out.write(",{}".format(100 * iou_data[i_row, i_col]))
        f_out.write("\n")
