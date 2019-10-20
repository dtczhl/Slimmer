"""
    Plot Labels
"""

import numpy as np
import pptk

data = np.loadtxt("/home/dtc/MyGit/dtc-scannet-sparseconvnet/Cpp/add_label/build/abc.ply")

valid_index = (data[:, 6] >= 0) & (data[:, 6] <= 19)
coords = data[valid_index, :3]
colors = data[valid_index, 3:6] /255.0
orig_label = data[valid_index, 6]
pred_label = data[valid_index, 7]

pptk.viewer(coords, colors)

