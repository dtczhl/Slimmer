"""
    sparsity of ScanNet
"""

import numpy as np
import torch
import glob
import os
import sys

scannet_dir = "/home/dtc/Data/ScanNet"

cell_size_arr = np.arange(0.03, 0.21, 0.02)

# path to pth
original_dir = os.path.join(scannet_dir, "Pth/Original")
save_dir = "../Result/EmptyCell"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

empty_filename = "scannet_empty_cell.csv"
save_file = os.path.join(save_dir, empty_filename)

pth_files = glob.glob(os.path.join(original_dir, "*.pth"))

n_pth_file = 100
pth_files = pth_files[:n_pth_file]

result = []

for cell_size in cell_size_arr:

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

print("saving file to:", save_file)
np.savetxt(save_file, result, fmt="%.3f,%.2f", header="cell_size,empty_ratio(%)")


