"""
    show results
"""

import torch
import numpy as np
import sys
import os
import pptk
import iou

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture']
CLASS_COLOR = [
    [138, 43, 226], [0, 128, 128], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [0, 255, 255], [255, 0, 255], [192, 192, 192], [128, 128, 128], [128, 0, 0],
    [128, 128, 0], [0, 128, 0], [128, 0, 128], [255, 0, 0], [0, 0, 128],
    [34, 139, 34], [64, 224, 208], [0, 0, 0], [75, 0, 130], [205, 133, 63]
]
CLASS_COLOR = np.array(CLASS_COLOR) / 255.0


def show_raw(raw_dir, raw_file):
    data = torch.load(os.path.join(raw_dir, raw_file))
    coords, colors, labels = data
    coords = np.array(coords, "float32")
    # colors = np.array(colors, "float32")
    labels = np.array(labels, "float32")
    ignore_index = labels == -100
    coords = coords[~ignore_index]
    colors = colors[~ignore_index]
    labels = labels[~ignore_index]
    gt_color = [CLASS_COLOR[x] for x in labels.astype("int32")]
    v1 = pptk.viewer(coords, (colors+1)/2)
    v2 = pptk.viewer(coords, gt_color)
    v1.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
    v2.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)


def show_result(result_dir, result_id):

    valOffsets_file = os.path.join(result_dir, "valOffsets.npy")
    valOffsets = np.load(valOffsets_file)

    data_file = os.path.join(result_dir, "data.npy")
    results = np.load(data_file)

    ignore_index = results[:, 6] == -100
    results = results[~ignore_index]
    n_tot = len(results)
    n_correct = np.sum(results[:, 6] == results[:, 7])
    print("Total pixel accuracy:", n_correct / n_tot)

    if result_id >= len(valOffsets)-1:
        sys.exit("id: " + result_id + " overflow")

    start_index, end_index = valOffsets[result_id], valOffsets[result_id+1]
    coords = results[start_index:end_index, :3]
    gt = results[start_index:end_index, 6]
    pred = results[start_index:end_index, 7]

    gt_color = [CLASS_COLOR[x] for x in gt.astype("int32")]
    pred_color = [CLASS_COLOR[x] for x in pred.astype("int32")]

    print("Pixel Accuracy for", result_id, "---", np.sum(gt == pred) / len(gt))

    pptk.viewer(coords, gt_color)
    pptk.viewer(coords, pred_color)


# scene0011_00_vh_clean_2.pth, "scene0015_00_vh_clean_2.pth"
show_raw("/home/dtc/Data/ScanNet/Pth/Original", "scene0011_00_vh_clean_2.pth")
#show_result("/home/dtc/Data/ScanNet/Accuracy/unet_scale100_m16_rep2_residualTrue-000000220/Curvature/10", 1)
