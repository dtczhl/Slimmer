"""
    Processing time
"""
import torch
import glob
import os
import numpy as np
import sys
import time
from shutil import copyfile, rmtree

# ------ configuration ------

scannet_dir = "/home/dtc/Data/ScanNet"

keep_ratio_arr = range(2, 101, 2)

# --- end of configuration

data_type = "Random"

tmp_dir = "../tmp/"

# in microseconds
time_file = "time.txt"

rmtree(tmp_dir)
os.makedirs(tmp_dir)

dst_dir = os.path.join(tmp_dir, "junk")
os.makedirs(dst_dir)

pth_original_dir = os.path.join(scannet_dir, "Pth/Original")
files = sorted(glob.glob(os.path.join(pth_original_dir, "*.pth")))

for keep_ratio in keep_ratio_arr:
    start_time = time.time()
    for src_file in files:
        data = torch.load(src_file)
        coords, colors, labels = data
        coords = np.array(coords, "float32")
        colors = np.array(colors, "float32")
        labels = np.array(labels, "float32")
        original_data = np.c_[coords, colors, labels]
        src_filename = os.path.basename(src_file)
        tmp_file_name = os.path.join(tmp_dir, src_filename)
        original_data.ravel().tofile(tmp_file_name)

        mycmd = "../Cpp/sample_data/build/sample_data {} {} {} {}" \
            .format(data_type.lower(), dst_dir, src_filename, keep_ratio)
        os.system(mycmd)
        os.remove(tmp_file_name)

    save_time_dir = os.path.join(tmp_dir, "time")
    if not os.path.exists(save_time_dir):
        os.makedirs(save_time_dir)

    copyfile(os.path.join(tmp_dir, time_file), os.path.join(save_time_dir, "{}.{}".format(time_file, keep_ratio)))
    print("keep ratio:", keep_ratio, "{:.2f}s".format(time.time() - start_time))
