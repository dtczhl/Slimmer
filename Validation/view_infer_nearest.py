"""
    View inferred labels. Perform difference between pth_first_file and pth_second_file

    generating a ply
"""

import torch
import numpy as np
from plyfile import PlyData, PlyElement

pth_first_file = "/home/dtc/Backup/Data/ScanNet/Pth/Random/100/scene0011_00_vh_clean_2.pth"

pth_second_file = "/home/dtc/Backup/Data/ScanNet/Pth/Random/40/scene0011_00_vh_clean_2.pth"

dst_path_file = "/home/dtc/Desktop/infer_diff.ply"


data_first = torch.load(pth_first_file)
coord_first, color_first, label_first = data_first

data_second = torch.load(pth_second_file)
coord_second, color_second, label_second = data_second

data_save_coord = []
data_save_color = []
data_save_label = []

for i_first in range(len(coord_first)):
    print(i_first, len(coord_first))
    min_dist = 1E9
    min_index = -1
    for i_second in range(len(coord_second)):
        dist = (coord_first[i_first][0] - coord_second[i_second][0])**2 + \
               (coord_first[i_first][1] - coord_second[i_second][1])**2 + \
               (coord_first[i_first][2] - coord_second[i_second][2])**2
        if dist < min_dist:
            min_dist = dist
            min_index = i_second
    if min_dist > 0:
        data_save_coord.append(coord_second[min_index])
        data_save_color.append(color_second[min_index])
        data_save_label.append(label_second[min_index])


print("Points:", [len(coord_first), len(coord_second)])
print(len(data_save_coord))

ply_save = []
for i in range(len(data_save_coord)):
    ply_save.append((data_save_coord[i][0], data_save_coord[i][1], data_save_coord[i][2],
                     data_save_color[i][0], data_save_color[i][1], data_save_color[i][2], data_save_label[i]))
ply_save = np.array(ply_save,
                    dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                           ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                           ("label", "u1")])
el = PlyElement.describe(ply_save, "vertex")
plydata = PlyData([el], text=True)
plydata.write(dst_path_file)
print("saving file to ", dst_path_file)




