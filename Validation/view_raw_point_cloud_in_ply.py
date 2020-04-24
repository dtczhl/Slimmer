"""
    Showing raw point cloud in ply
"""

import numpy as np
import torch
import pptk
from plyfile import PlyData

# --- Path to the ply_file
ply_file = "/home/dtc/Backup/Data/ScanNet/Ply/scene0015_00_vh_clean_2.ply"
# ply_file = "/home/dtc/Desktop/train_ply/Ply/scene0050_00_vh_clean_2.ply"


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


ply_data = PlyData.read(ply_file)
data_xyz = np.asarray([ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]).transpose() # x, y, z for each point
data_rgb = np.asarray([ply_data["vertex"]["red"], ply_data["vertex"]["green"], ply_data["vertex"]["blue"]]).transpose() # r, g, b for each point
data_rgb = data_rgb / 255.0  # color normalized to [0, 1.0]
data_label = np.asarray(ply_data["vertex"]["label"]).transpose()

ignore_index = (0 > data_label) | (data_label > 19)
data_xyz = data_xyz[~ignore_index]
data_rgb = data_rgb[~ignore_index]
data_label = data_label[~ignore_index]
# print(data_label)
gt_color = [CLASS_COLOR[x] for x in data_label.astype("int32")]

# pptk.viewer(data_xyz, gt_color)
v = pptk.viewer(data_xyz, data_rgb)
v.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False,
       show_info=False)

