"""
    Showing raw point cloud in pth
"""

import numpy as np
import torch
import pptk

# pth_file = "/home/dtc/Data/ScanNet/Pth/Hierarchy/32/scene0011_00_vh_clean_2.pth"
pth_file = "/home/dtc/Backup/Data/ScanNet/Pth/Random/20/scene0011_00_vh_clean_2.pth"


data = torch.load(pth_file)
coords, colors, labels = data
coords = np.array(coords, "float32")
colors = np.array(colors, "float32")
labels = np.array(labels, "float32")
print(len(labels))

v = pptk.viewer(coords, (colors + 1)/2)
v.set(point_size=0.01, bg_color=[1, 1, 1, 1], floor_color=[1, 1, 1, 1], show_grid=False, show_axis=False, show_info=False)