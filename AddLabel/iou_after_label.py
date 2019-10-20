"""
    Calculate IOU
"""

import numpy as np
import glob
import os
import iou

txt_dir = "/home/dtc/Backup/Data/ScanNet/PlyLabelAdd/Random/40"

txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))

gt = []
pred = []
for txt_file in txt_files:
    print(txt_file)
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
