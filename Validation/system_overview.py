"""
    System overview. For paper
"""

import numpy as np
import torch
import pptk
import random


# ------ Configuration -------
keep_ratio = 40

# --- end of Configuration ---


# pth_file = "/home/dtc/Data/ScanNet/Pth/Hierarchy/32/scene0011_00_vh_clean_2.pth"
pth_file = "/home/dtc/Backup/Data/ScanNet/Pth/Random/100/scene0011_00_vh_clean_2.pth"


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


data = torch.load(pth_file)
coords, colors, labels = data
coords = np.array(coords, "float32")
colors = np.array(colors, "float32")
labels = np.array(labels, "float32")

ignore_index = np.logical_or(labels < 0, labels >= 20)
coords = coords[~ignore_index]
colors = colors[~ignore_index]
labels = labels[~ignore_index]

# plot original pc
v = pptk.viewer(coords, (colors + 1)/2)
v.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
v.set(lookat=[0, 0, 0], phi=-0.1)

idx_keep = np.random.rand(len(coords)) < keep_ratio / 100

coords_keep = coords[idx_keep, :]
colors_keep = colors[idx_keep, :]
labels_keep = labels[idx_keep]

coords_remove = coords[~idx_keep, :]
colors_remove = colors[~idx_keep, :]
labels_remove = labels[~idx_keep]

# plot simplified pc
v = pptk.viewer(coords_keep, (colors_keep + 1)/2)
v.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
v.set(lookat=[0, 0, 0], phi=-0.1)

# plot labels of points that are kept
label_color_keep = [CLASS_COLOR[x] for x in labels_keep.astype("int32")]
v = pptk.viewer(coords_keep, label_color_keep)
v.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
v.set(lookat=[0, 0, 0], phi=-0.1)

# plot labels of points that are removed
label_color_remove = [CLASS_COLOR[x] for x in labels_remove.astype("int32")]
v = pptk.viewer(coords_remove, label_color_remove)
v.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
v.set(lookat=[0, 0, 0], phi=-0.1)

# plot labels of original pc
label_color = [CLASS_COLOR[x] for x in labels.astype("int32")]
v = pptk.viewer(coords, label_color)
v.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)
v.set(lookat=[0, 0, 0], phi=-0.1)