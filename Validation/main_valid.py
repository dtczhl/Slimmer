"""
    Validating IOU of a trained model.

    Configurations
        scannet_dir: root dir of scannet
        device: for saving
        model_name: trained DNN
        data_type: Random, Grid, Hierarchy
        specify_id: if want to valid specific ids
        n_scene: how many scenes to validate

    Results saved to ../Result/{device}/{model_name}/{data_type}/result_main.csv.
        Format: keep ratio; average number of points; mean IOU; average running time (s); FLOPs (M); memory (M)

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

# ------ Configurations ------
scannet_dir = "/home/dtc/Backup/Data/ScanNet"

device = "alienware"

# trained model in ../Model/
model_name = "scannet_m32_rep2_residualTrue-000000670.pth"

# Random, Grid, Hierarchy
data_type = "Random"

specify_id = []  # if want to valid specific ids

use_cuda = True

#!!!!!!!!!!!
n_scene = 50

# --- end of configurations ---

# Model Options
extract_model_options = model_name.split("-")[0]
extract_model_options = extract_model_options.split("_")
m = int(extract_model_options[1][1:])
block_reps = int(extract_model_options[2][3:])
residual_blocks = extract_model_options[3][8:]
if residual_blocks == "True":
    residual_blocks = True
elif residual_blocks == "False":
    residual_blocks = False
else:
    sys.exit("Unknown residual blocks")

# m = 16  # 16 or 32; 16
# residual_blocks = True  # True or False; False
# block_reps = 2  # Conv block repetition factor: 1 or 2; 1

dimension = 3
scale = 100
full_scale = 4096

offset_filename = "valOffsets.npy"
result_filename = "data.npy"

val = []
valOffsets = []
valLabels = []


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


print(" --- loading model ---", model_name)
model_file = os.path.join("../Model", model_name)
unet = Model()
unet.load_state_dict(torch.load(model_file))
if use_cuda:
    unet.cuda()

save_dir = os.path.join("../Result", device, os.path.splitext(model_name)[0], data_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


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


def valid_data(data_id):
    # data_id is ratio of point clouds

    start_time = time.time()

    process = psutil.Process(os.getpid())
    ret_memory = 0
    ret_time = 0

    data_name = data_type + "/" + str(data_id)

    ret_data_id = data_id

    data_dir = os.path.join(scannet_dir, "Pth", data_name)

    # load val data
    print("loading val data", data_name)
    global val
    val = []
    if "n_scene" in locals() or "n_scene" in globals():
        for x in torch.utils.data.DataLoader(
                glob.glob(os.path.join(data_dir, "*.pth"))[:n_scene],
                collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
            val.append(x)
    else:
        for x in torch.utils.data.DataLoader(
                glob.glob(os.path.join(data_dir, "*.pth")),
                collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
            val.append(x)
    print("data from {} scenes".format(len(val)))

    global valOffsets
    global valLabels
    valOffsets = [0]
    valLabels = []
    for idx, x in enumerate(val):
        valOffsets.append(valOffsets[-1]+x[2].size)
        valLabels.append(x[2].astype(np.int32))
    valLabels = np.hstack(valLabels)

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

            start_time_ret = time.time()
            predictions = unet(batch['x'])
            store.index_add_(0, batch['point_ids'], predictions.cpu())
            ret_memory += process.memory_info().rss / 1e6
            ret_time += time.time() - start_time_ret

        ret_muladd = scn.forward_pass_multiplyAdd_count / 1e6
        print('Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count / len(val) / 1e6, 'MegaHidden',
              scn.forward_pass_hidden_states / len(val) / 1e6, 'time=', time.time() - start, 's',
              "Memory (M)=", ret_memory / len(val))
        ret_iou = iou.evaluate(store.max(1)[1].numpy(), valLabels)

        print("Time for data_id {}: {:.2f} s".format(data_id, time.time() - start_time))

        return ret_data_id, len(valLabels)/len(val), 100*ret_iou, ret_time/len(val), ret_muladd/len(val), ret_memory/len(val)


if __name__ == "__main__":
    result = []

    def func_filename(x):
        return int(os.path.basename(x))

    if specify_id:
        for my_id in specify_id:
            result.append(valid_data(my_id))
    else:
        data_dirs = sorted(glob.glob(os.path.join(scannet_dir, "Pth", data_type, "*")), key=func_filename)
        for data_dir in data_dirs:
            my_id = int(os.path.basename(data_dir))
            result.append(valid_data(my_id))

    result_vstack = np.vstack(result)
    print("id, avg num of points, mean iou, avg time (s), avg_flop(M), memory(M)")
    print(np.array_str(result_vstack, precision=2, suppress_small=True))

    save_file = os.path.join(save_dir, "result_main.csv")
    print("saving file to:", save_file)
    np.savetxt(save_file, result, fmt="%d,%.2f,%.2f,%.2f,%.2f,%.2f",
               header="data_id,avg_num_points,mean_iou,avg_time(s),avg_addmul(M),memory(M)")
