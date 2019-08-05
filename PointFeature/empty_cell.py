"""
    sparsity of ScanNet
"""

import numpy as np
import torch
import glob
import os
import sys

scannet_dir = "/home/dtc/Data/ScanNet"

cell_size_arr = np.arange(0.01, 0.11, 0.01)

# path to pth
original_dir = os.path.join(scannet_dir, "Pth/Original")
result_dir = os.path.join(scannet_dir, "Result")

empty_filename = "empty.csv"
result_file = os.path.join(result_dir, empty_filename)
print("result file:", result_file)

pth_files = glob.glob(os.path.join(original_dir, "*.pth"))

result = []

for cell_size in cell_size_arr:

    n_pth_file = len(pth_files)
    ratio_empty_tot = 0.0
    for pth_file in pth_files:
        print("cell size:", cell_size, pth_file)
        data = torch.load(pth_file)
        coords, colors, labels = data

        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        n_coords = (max_coords - min_coords) // cell_size + 1
        n_coords = n_coords.astype("int32")
        grid = np.zeros((n_coords[0], n_coords[1], n_coords[2]))

        for point in coords:
            point_coords = (point - min_coords) // cell_size
            point_coords = point_coords.astype("int32")
            grid[point_coords[0], point_coords[1], point_coords[2]] = 1

        ratio_empty_tot += (np.prod(grid.shape) - np.sum(grid)) / np.prod(grid.shape) * 100
    result_item = [cell_size, ratio_empty_tot/n_pth_file]
    print("------", result_item)
    result.append(result_item)

result = np.vstack(result)
print(np.array_str(result, precision=2, suppress_small=True))

np.savetxt(result_file, result, fmt="%.3f,%.2f", header="cell_size,empty_ratio(%)")


