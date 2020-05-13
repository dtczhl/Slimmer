"""
    Sparsity of KITTI
"""

import numpy as np
import torch
import glob
import os
import sys

data_dir = "/home/dtc/Data/KITTI"

# cell_size_arr = np.arange(0.03, 0.21, 0.02)
cell_size_arr = [0.1]

original_dir = os.path.join(data_dir, "PointPillars/testing/velodyne_reduced")
save_dir = "../Result/EmptyCell"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

empty_filename = "kitti_empty_cell.csv"
save_file = os.path.join(save_dir, empty_filename)

bin_files = glob.glob(os.path.join(original_dir, "*.bin"))

n_file = 100  # calculate n_file only
bin_files = bin_files[:n_file]

result = []

for cell_size in cell_size_arr:

    ratio_empty_tot = 0.0
    for bin_file in bin_files:
        print("cell size:", cell_size, bin_file)
        data = np.fromfile(bin_file, "<f4")
        data = np.reshape(data, (-1, 4))
        coords = data[:, :3]

        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        n_coords = (max_coords - min_coords) // cell_size + 1
        n_coords = n_coords.astype("int32")
        #print(n_coords)
        # sys.exit()
        grid = np.zeros((n_coords[0], n_coords[1], n_coords[2]))

        for point in coords:
            point_coords = (point - min_coords) // cell_size
            point_coords = point_coords.astype("int32")
            grid[point_coords[0], point_coords[1], point_coords[2]] = 1

        ratio_empty_tot += (np.prod(grid.shape) - np.sum(grid)) / np.prod(grid.shape) * 100
    result_item = [cell_size, ratio_empty_tot/n_file]
    print("------", result_item)
    result.append(result_item)

result = np.vstack(result)
print(np.array_str(result, precision=2, suppress_small=True))

print("saving file to:", save_file)
np.savetxt(save_file, result, fmt="%.3f,%.2f", header="cell_size,empty_ratio(%)")
