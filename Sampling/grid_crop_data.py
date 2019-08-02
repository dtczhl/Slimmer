"""
    Crop data with Menger curvature sampling
"""
import torch
import glob
import os
import numpy as np
import sys
import time
import pptk

scannet_dir = "/home/dtc/Data/ScanNet"
data_type = "Grid"

# 0.1: 4.3%
# 0.01: 96.9%
cell_size_arr = np.linspace(0.01, 0.1, 100)


def crop_data(cell_size):

    start_time = time.time()

    # clear tmp
    files = glob.glob("../tmp/*")
    for file in files:
        os.remove(file)

    original_dir = os.path.join(scannet_dir, "Pth/Original")
    dst_dir = os.path.join(scannet_dir, "Pth/{}".format(data_type))

    original_number_points_tot = 0
    trim_number_points_tot = 0

    files = sorted(glob.glob(os.path.join(original_dir, '*.pth')))
    for src_file in files:
        src_filename = os.path.basename(src_file)
        data = torch.load(src_file)
        coords, colors, labels = data
        coords = np.array(coords, "float32")
        colors = np.array(colors, "float32")
        labels = np.array(labels, "float32")

        original_number_points_tot += len(labels)

        # copy file
        original_data = np.c_[coords, colors, labels]
        tmp_dir = "../tmp/"
        tmp_file_name = os.path.join(tmp_dir, src_filename)
        original_data.ravel().tofile(tmp_file_name)

        mycmd = "../Cpp/sample_data/build/sample_data {} {} {} {}"\
            .format(data_type.lower(), dst_dir, src_filename, cell_size)
        os.system(mycmd)
        os.remove(tmp_file_name)

        src_trim_file = tmp_file_name + ".trim"
        if not os.path.exists(src_trim_file):
            sys.exit("Error, file " + src_trim_file + " does not exist")

        # calculate number of points
        new_data = np.fromfile(src_trim_file, "<f4")
        new_data = np.reshape(new_data, (-1, 7))
        new_coords = new_data[:, :3]
        new_colors = new_data[:, 3:6]
        new_labels = new_data[:, 6]

        trim_number_points_tot += len(new_labels)

    ratio_point = int(trim_number_points_tot/original_number_points_tot*100)

    if not os.path.exists(os.path.join(dst_dir, "{}".format(ratio_point))):
        os.makedirs(os.path.join(dst_dir, "{}".format(ratio_point)))

    # copy files to dst
    files = sorted(glob.glob(os.path.join(tmp_dir, '*.trim')))
    for trim_file in files:
        src_filename = os.path.basename(trim_file)
        src_filename = src_filename[:-5]  # remove .trim
        new_data = np.fromfile(trim_file, "<f4")
        new_data = np.reshape(new_data, (-1, 7))
        os.remove(trim_file)

        new_coords = new_data[:, :3]
        new_colors = new_data[:, 3:6]
        new_labels = new_data[:, 6]

        new_coords = np.array(new_coords, "float32")
        new_colors = np.array(new_colors, "float32")
        new_labels = np.array(new_labels, "float32")

        dst_file_path = os.path.join(dst_dir, "{}/{}".format(ratio_point, src_filename))
        # print(trim_file, " ---> ", dst_file_path)
        new_coords = np.ascontiguousarray(new_coords)
        new_colors = np.ascontiguousarray(new_colors)
        new_labels = np.ascontiguousarray(new_labels)
        torch.save((new_coords, new_colors, new_labels), dst_file_path)

    print("------ cell_size {:.3f}, ratio {}%, {:.2f}s".format(cell_size, ratio_point, time.time() - start_time))


if __name__ == "__main__":
    for cell_size in cell_size_arr:
        crop_data(cell_size)
