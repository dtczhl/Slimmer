"""
    view predication for point cloud
"""

import torch
import numpy as np
import sys
import os
import pptk

# ------ Configurations ------

# path to pth file
pth_file = "../tmp/scene0011_00_vh_clean_2.pth.Random.10"

show_gt = False  # show groundtruth or not

# --- end of configurations ---

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


def show_predication_result(pth_file, show_gt):

    data = torch.load(pth_file)
    coords, colors, labels, pred = data

    ignore_index = labels == -100
    coords = coords[~ignore_index]
    colors = colors[~ignore_index]
    labels = labels[~ignore_index]
    pred = pred[~ignore_index]

    gt_color = [CLASS_COLOR[x] for x in labels.astype("int32")]
    pred_color = [CLASS_COLOR[x] for x in pred.astype("int32")]

    if show_gt:
        v1 = pptk.viewer(coords, gt_color)
        v1.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)

    v2 = pptk.viewer(coords, pred_color)
    v2.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)


if __name__ == "__main__":
    show_predication_result(pth_file, show_gt)
