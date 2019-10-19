"""
    add label to missing points
"""

import numpy as np
import torch
import os
import pandas as pd

original_pth_dir = "/home/dtc/Backup/Data/ScanNet/Pth/Original"
simplified_pth_dir = "/home/dtc/Backup/Data/ScanNet/PthLabel/Random/98"

pth_file = "scene0011_00_vh_clean_2.pth"

original_pth_file = os.path.join(original_pth_dir, pth_file)
simplified_pth_file = os.path.join(simplified_pth_dir, pth_file)

original_data = torch.load(original_pth_file)
original_coords, original_colors, original_labels = original_data
coords = np.array(original_coords, "float32")
colors = np.array(original_colors, "float32")
labels = np.array(original_labels, "float32")
original_data = np.c_[coords, colors, labels]

simplified_data = torch.load(simplified_pth_file)
simplified_coords, simplified_colors, simplified_labels = simplified_data
coords = np.array(simplified_coords, "float32")
colors = np.array(simplified_colors, "float32")
labels = np.array(simplified_labels, "float32")
simplified_data = np.c_[coords, colors, labels]

print(len(original_data))
n_match = 0
n_dismatch = 0
for row_original in range(len(original_data)):
    x_ori, y_ori, z_ori, r_ori, g_ori, b_ori, l_ori = original_data[row_original]
    is_match = False
    for row_simplified in range(len(simplified_data)):
        x_sim, y_sim, z_sim, r_smi, g_smi, b_smi, l_smi = simplified_data[row_simplified]
        eps = 1E-6
        if abs(x_ori - x_sim) < eps and abs(y_ori - y_sim) < eps and abs(z_ori - z_sim) < eps:
            # find point
            original_data[row_original][6] = l_smi
            is_match = True
            n_match += 1
            break
    # not find point, get label from nearest point
    if not is_match:
        min_dist = 1E10
        for row_simplified in range(len(simplified_data)):
            x_sim, y_sim, z_sim, r_smi, g_smi, b_smi, l_smi = simplified_data[row_simplified]
            if (x_ori - x_sim)**2 + (y_ori - y_sim)**2 + (z_ori - z_sim)**2 < min_dist:
                min_dist = (x_ori - x_sim)**2 + (y_ori - y_sim)**2 + (z_ori - z_sim)**2
                original_data[row_original][6] = l_smi
        n_dismatch += 1

print([n_match, n_dismatch])



