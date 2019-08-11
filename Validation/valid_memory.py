"""
    validating memory
"""

import torch
import torch.nn as nn
import numpy as np
import glob
import math

import sparseconvnet as scn
import iou

import torch.utils.data
import multiprocessing as mp
import os
import time
import sys
import psutil
import pptk

# ------ Configuration ------

scannnet_dir = '/home/dtc/Data/ScanNet'

model_file = '/home/dtc/Data/ScanNet/Model/scannet_m16_rep2_residualTrue-000000530.pth'

use_cuda = False

# Original, Random, Grid, Hierarchy
data_type = "Original"

keep_ratio = 42

filename = 'scene0011_00_vh_clean_2.pth'

# --- end of configuration ---


if data_type.lower() == "original":
    target_file = os.path.join(scannnet_dir, 'Pth', data_type, filename)
else:
    target_file = os.path.join(scannnet_dir, "Pth", data_type, "{}".format(keep_ratio), filename)

tmp_dir = '../tmp'

# Model Options
m = 16  # 16 or 32; 16
residual_blocks = True  # True or False; False
block_reps = 2  # Conv block repetition factor: 1 or 2; 1

dimension = 3
scale = 100
full_scale = 4096


# load model
class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
               scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, 20)

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x


def coords_transform(physical_val):
    a, b, c = physical_val
    m = np.eye(3)
    m *= scale
    # theta = np.random.rand()*2*math.pi
    theta = 0
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    a = np.matmul(a, m) + full_scale / 2
    m = a.min(0)
    M = a.max(0)
    q = M - m
    offset = -m
    a += offset
    idxs = (a.min(1) >= 0) * (a.max(1) < full_scale)
    a = a[idxs]
    b = b[idxs]
    c = c[idxs]
    return a, b, c


print(" --- loading model ---")
unet = Model()
unet.load_state_dict(torch.load(model_file))
if use_cuda:
    unet.cuda()
unet.eval()


def valid_file(f_pth):
    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0

    data = torch.load(f_pth)
    coords, colors, label = coords_transform(data)
    coords = torch.from_numpy(coords).long()
    coords = torch.cat([coords, torch.LongTensor(coords.shape[0], 1).fill_(0)], 1)
    colors = torch.from_numpy(colors)

    if use_cuda:
        coords = coords.cuda()
        colors = colors.cuda()

    y = unet([coords, colors])
    y = y.cpu().detach().numpy()
    y = np.argmax(y, axis=1)

    if data_type.lower() == "original":
        save_file = os.path.join(tmp_dir, "{}.{}".format(os.path.basename(f_pth), data_type))
    else:
        save_file = os.path.join(tmp_dir, "{}.{}.{}".format(os.path.basename(f_pth), data_type, keep_ratio))

    print("saving file to:", save_file)
    torch.save((data[0], data[1], data[2], y), save_file)


if __name__ == "__main__":
    valid_file(target_file)
