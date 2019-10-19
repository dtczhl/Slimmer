"""
    plot labels
"""

import numpy as np
import torch

pth_file = "/home/dtc/Backup/Data/ScanNet/Pth/Random/100/scene0423_01_vh_clean_2.pth"

data = torch.load(pth_file)
coords, colors, labels = data
coords = np.array(coords, "float32")
colors = np.array(colors, "float32")
labels = np.array(labels, "float32")
print(labels)