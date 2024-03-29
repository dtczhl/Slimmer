"""
    Plot point clouds after KNN
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
import pptk

# ------ Configurations ------

# path to txt file
txt_file = "/home/dtc/Backup/Data/ScanNet/AddMissingLabel/Grid/1/20/scene0015_00_vh_clean_2.txt"

show_gt = False  # show groundtruth or not; groudtruth draw first, i.e., on back

# --- end of configurations ---

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture']
# CLASS_COLOR = [
#     [138, 43, 226], [0, 128, 128], [0, 255, 0], [0, 0, 255], [255, 255, 0],
#     [0, 255, 255], [255, 0, 255], [192, 192, 192], [128, 128, 128], [128, 0, 0],
#     [128, 128, 0], [0, 128, 0], [128, 0, 128], [255, 0, 0], [0, 0, 128],
#     [34, 139, 34], [64, 224, 208], [0, 0, 0], [75, 0, 130], [205, 133, 63]
# ]
SCANNET_COLOR_MAP = SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

VALID_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]

CLASS_COLOR = []
for valid_id in VALID_CLASS_IDS:
    CLASS_COLOR.append(SCANNET_COLOR_MAP[valid_id])
CLASS_COLOR = np.array(CLASS_COLOR) / 255.0


def show_predication_result(txt_file, show_gt):

    data_pd = pd.read_csv(txt_file, sep=" ", header=None,
                          names=["x", "y", "z", "r", "g", "b", "label_orig", "label_pred"])
    # data = torch.load(pth_file)
    coords = data_pd[["x", "y", "z"]].to_numpy()
    colors = data_pd[["r", "g", "b"]].to_numpy()
    labels = data_pd["label_orig"].to_numpy()
    pred = data_pd["label_pred"].to_numpy()

    valid_index = (0 <= labels) & (labels <= 19)
    coords = coords[valid_index]
    colors = colors[valid_index]
    labels = labels[valid_index]
    pred = pred[valid_index]

    print(np.unique(pred))

    gt_color = [CLASS_COLOR[x] for x in labels.astype("int32")]
    pred_color = [CLASS_COLOR[x] for x in pred.astype("int32")]

    if show_gt:
        v1 = pptk.viewer(coords, gt_color)
        v1.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
        v1.set(theta=1.8, lookat=[0, 0, 0], phi=0.52)

    v2 = pptk.viewer(coords, pred_color)
    v2.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
    v2.set(theta=1.8, lookat=[0, 0, 0], phi=0.52)

if __name__ == "__main__":
    show_predication_result(txt_file, show_gt)
