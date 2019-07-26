"""
    show results
"""

import numpy as np
import sys
import os
import pptk
import iou

result_dir = "/home/dtc/Data/ScanNet/Accuracy/unet_scale100_m16_rep2_residualTrue-000000064-unet/Original"

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
CLASS_COLOR = [
    [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [0, 255, 255], [255, 0, 255], [192, 192, 192], [128, 128, 128], [128, 0, 0],
    [128, 128, 0], [0, 128, 0], [128, 0, 128], [0, 128, 128], [0, 0, 128],
    [34, 139, 34], [64, 224, 208], [138, 43, 226], [75, 0, 130], [205, 133, 63]
]
CLASS_COLOR = np.array(CLASS_COLOR) / 255.0

valOffsets_file = os.path.join(result_dir, "valOffsets.txt")
valOffsets = np.loadtxt(valOffsets_file, dtype="uint64")

data_file = os.path.join(result_dir, "data.txt")
results = np.loadtxt(data_file, dtype="float32")

n_tot = len(results)
n_correct = np.sum(results[:, 6] == results[:, 7])
print("Total accuracy:", n_correct/n_tot)

# index_ignore = results[:, 6] == -100
# coords = results[:, :3][~index_ignore]
# gt = results[:, 6][~index_ignore]
# pred = results[:, 7][~index_ignore]
# n_correct = np.sum(gt == pred)
# n_tot = len(gt)
# print("Accuracy with ignores:", n_correct/n_tot)


def show(result_id):
    if result_id >= len(valOffsets)-1:
        sys.exit("id: " + result_id + " overflow")

    start_index, end_index = valOffsets[result_id], valOffsets[result_id+1]
    coords = results[start_index:end_index, :3]
    gt = results[start_index:end_index, 6]
    pred = results[start_index:end_index, 7]

    ignore_index = gt == -100
    coords = coords[~ignore_index]
    gt = gt[~ignore_index]
    pred = pred[~ignore_index]

    gt_color = [CLASS_COLOR[x] for x in gt.astype("int32")]
    pred_color = [CLASS_COLOR[x] for x in pred.astype("int32")]

    print("Accuracy with ignores",  np.sum(gt == pred) / len(gt))

    pptk.viewer(coords, gt_color)
    pptk.viewer(coords, pred_color)


show(0)
