import torch
import torch.nn as nn
import numpy as np
import glob
import math

import sparseconvnet as scn
import iou

import torch.utils.data
import scipy.ndimage
import multiprocessing as mp
import os
import time
import sys

# ------ Configurations ------
scannet_dir = "/home/dtc/Data/ScanNet"

model_name = "unet_scale100_m16_rep2_residualTrue-000000064-unet.pth"

data_name = "Original"

use_cuda = True

dimension = 3
scale = 100
full_scale = 4096

# Options
m = 16  # 16 or 32; 16
residual_blocks = True  # True or False; False
block_reps = 2  # Conv block repetition factor: 1 or 2; 1

# --- end of configurations ---

data_dir = os.path.join(scannet_dir, "Pth", data_name)
model_file = os.path.join(scannet_dir, "Model", model_name)

save_dir = os.path.join(scannet_dir, "Accuracy", os.path.splitext(model_name)[0], data_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

offset_filename = "valOffsets.txt"
result_filename = "data.txt"

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


print("loading model")
unet = Model()
unet.load_state_dict(torch.load(model_file))
if use_cuda:
    unet.cuda()

# load val data
print("loading val data")
val = []
for x in torch.utils.data.DataLoader(
        glob.glob(os.path.join(data_dir, "*.pth")),
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    val.append(x)
print("data from {} scenes".format(len(val)))

valOffsets = [0]
valLabels = []
for idx, x in enumerate(val):
    valOffsets.append(valOffsets[-1]+x[2].size)
    valLabels.append(x[2].astype(np.int32))
valLabels = np.hstack(valLabels)


def valMerge(tbl):
    locs = []
    feats = []
    labels = []
    point_ids = []
    for idx, i in enumerate(tbl):
        a, b, c = val[i]
        m = np.eye(3)
        m *= scale
        # theta = np.random.rand()*2*math.pi
        theta = 0
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m)+full_scale/2
        m = a.min(0)
        M = a.max(0)
        q = M-m
        offset = -m
        a += offset
        idxs = (a.min(1) >= 0)*(a.max(1) < full_scale)
        a = a[idxs]
        b = b[idxs]
        c = c[idxs]
        a = torch.from_numpy(a).long()
        locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    point_ids = torch.cat(point_ids, 0)
    return {'x': [locs, feats], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids}


val_data_loader = torch.utils.data.DataLoader(
    list(range(len(val))), batch_size=1, collate_fn=valMerge, num_workers=20, shuffle=False)

print("calculating accuracy ")
with torch.no_grad():
    unet.eval()
    store = torch.zeros(valOffsets[-1], 20)
    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0
    start = time.time()
    # for rep in range(1, 1 + val_reps):
    for i, batch in enumerate(val_data_loader):
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['y'] = batch['y'].cuda()
        predictions = unet(batch['x'])
        store.index_add_(0, batch['point_ids'], predictions.cpu())
    print('Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count / len(val) / 1e6, 'MegaHidden',
          scn.forward_pass_hidden_states / len(val) / 1e6, 'time=', time.time() - start, 's')
    iou.evaluate(store.max(1)[1].numpy(), valLabels)

    print("saving results")
    val_arr = []
    for scene in val:
        coords, colors, labels = scene
        var_scene = np.c_[coords, colors, labels]
        val_arr.append(var_scene)
    val_arr = np.vstack(val_arr)
    val_arr = np.c_[val_arr, store.max(1)[1].numpy()]
    np.savetxt(os.path.join(save_dir, offset_filename), valOffsets, fmt="%d")
    np.savetxt(os.path.join(save_dir, result_filename),  val_arr, fmt="%.2f %.2f %.2f %.2f %.2f %.2f %d %d")
