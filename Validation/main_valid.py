import torch
import torch.nn as nn
import numpy as np
import glob
import math
import sparseconvnet as scn
import torch.utils.data
import scipy.ndimage
import multiprocessing as mp
import os
import sys

model_path = ""

val_data_dir = "/home/dtc/Data/ScanNet/Pth/Original"

dimension = 3
scale = 100
full_scale = 4096

# Options
m = 16  # 16 or 32; 16
residual_blocks = True  # True or False; False
block_reps = 2  # Conv block repetition factor: 1 or 2; 1


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


use_cuda = False
unet = Model()
if scn.checkpoint_restore(unet, model_path, 'unet', use_cuda) <= 1:
    sys.exit("fails to load " + model_path)



sys.exit(0)

# load val data
val = []
for x in torch.utils.data.DataLoader(
        glob.glob(os.path.join(val_data_dir, "*.pth")),
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    val.append(x)

valOffsets = [0]
valLabels = []
for idx, x in enumerate(val):
    valOffsets.append(valOffsets[-1]+x[2].size)
    valLabels.append(x[2].astype(np.int32))
valLabels = np.hstack(valLabels)

