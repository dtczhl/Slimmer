import numpy as np
import torch

import pptk

pth_file = "/home/dtc/Data/ScanNet/Pth/Grid/4/scene0011_00_vh_clean_2.pth"

data = torch.load(pth_file)
coords, colors, labels = data
coords = np.array(coords, "float32")
colors = np.array(colors, "float32")
labels = np.array(labels, "float32")

pptk.viewer(coords)