"""
    Save to .bin from .pth

    Run after
        download_data.py
        split_data.py
        prepare_data.py
"""

import torch
import numpy as np
import pptk
import math
import os

# ------ Configurations ------

# path to ScanNet dataset
data_dir = "/home/dtc/Data/ScanNet"

# path to the processed pth folder
pth_dir = "/home/dtc/MyGit/dtc-sparseconvnet"

# path to save .bin
save_dir = "/home/dtc/Data/ScanNet/Bin/"

# --- end of Configurations ---


def pth_to_bin(pth_filename, bin_filename):
    data = torch.load(pth_filename)
    coords, colors, labels = data
    m = np.eye(3)
    theta = 0
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    coords = np.matmul(coords, m)
    bin_data = np.c_[coords, colors, labels]
    with open(bin_filename, "wb") as f_bin:
        bin_data.astype("float32").tofile(f_bin)


def read_bin(bin_filename):
    bin_data = np.fromfile(bin_filename, "<f4")
    bin_data = np.reshape(bin_data, (-1, 7))
    return bin_data


if __name__ == "__main__":

    data_train_txt = os.path.join(data_dir, "scannetv2_train.txt")
    data_valid_txt = os.path.join(data_dir, "scannetv2_val.txt")

    pth_train_dir = os.path.join(pth_dir, "train")
    pth_valid_dir = os.path.join(pth_dir, "val")

    with open(data_train_txt, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            line = line.strip()
            print(line)
            pth_filename = os.path.join(pth_train_dir, line + "_vh_clean_2.pth")
            bin_filename = os.path.join(os.path.join(save_dir, "train"), line + "_vh_clean_2.bin")
            pth_to_bin(pth_filename, bin_filename)

    with open(data_valid_txt, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            line = line.strip()
            print(line)
            pth_filename = os.path.join(pth_valid_dir, line + "_vh_clean_2.pth")
            bin_filename = os.path.join(os.path.join(save_dir, "val"), line + "_vh_clean_2.bin")
            pth_to_bin(pth_filename, bin_filename)
