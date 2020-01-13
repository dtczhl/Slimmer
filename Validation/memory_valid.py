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
import os
import time
import sys
import psutil
from plyfile import PlyData, PlyElement

# ------ Configuration ------

scannet_dir = '/home/dtc/Backup/Data/ScanNet'

device = "alienware"

model_name = 'scannet_m32_rep2_residualTrue-000000670.pth'

# Random, Grid, Hierarchy
data_type = "Random"

specify_id = []  # if want to valid specific ids

is_save_ply_label = False   # whether save prediction labels for each point

use_cuda = False

#!!!!!!!!!!!1
n_scene = 30

# --- end of configuration ---

model_file = os.path.join("../Model", model_name)

# Model Options
extract_model_options = model_name.split("-")[0]
extract_model_options = extract_model_options.split("_")
m = int(extract_model_options[1][1:])
block_reps = int(extract_model_options[2][3:])
block_reps = int(extract_model_options[2][3:])
residual_blocks = extract_model_options[3][8:]
if residual_blocks == "True":
    residual_blocks = True
elif residual_blocks == "False":
    residual_blocks = False
else:
    sys.exit("Unknown residual blocks")

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


print(" --- loading model ---", model_name)
unet = Model()
unet.load_state_dict(torch.load(model_file))
if use_cuda:
    unet.cuda()
unet.eval()

save_dir = os.path.join("../Result", device, os.path.splitext(model_name)[0], data_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def valid_data(data_id):

    start_time = time.time()

    ret_data_id = data_id

    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0

    process = psutil.Process(os.getpid())

    ret_memory = 0
    ret_time = 0
    ret_flop = 0

    pth_folder = os.path.join(scannet_dir, "Pth", data_type, str(data_id))
    pth_files = glob.glob(os.path.join(pth_folder, '*.pth'))

    # !!!!!!!!!!!!!!!!!!!!!!
    pth_files = pth_files[:n_scene]

    n_file = len(pth_files)

    for pth_file in pth_files:
        data = torch.load(pth_file)
        coords_bak, colors_bak, label_bak = data  # for saving purpose
        coords, colors, label = coords_transform(data)
        coords = torch.from_numpy(coords).long()
        coords = torch.cat([coords, torch.LongTensor(coords.shape[0], 1).fill_(0)], 1)
        colors = torch.from_numpy(colors)

        if use_cuda:
            coords = coords.cuda()
            colors = colors.cuda()

        start_time_ret = time.time()
        y = unet([coords, colors])
        y = y.cpu().detach().numpy()
        y = np.argmax(y, axis=1)

        ret_memory += process.memory_info().rss / 1e6
        ret_time += time.time() - start_time_ret

        # save pth and labels
        if is_save_ply_label:
            dst_pth_label_dir = os.path.join(scannet_dir, "PlyLabel", data_type, str(data_id))
            if not os.path.exists(dst_pth_label_dir):
                os.makedirs(dst_pth_label_dir)
            dst_path_file = os.path.join(dst_pth_label_dir, os.path.basename(pth_file)[:-4]+".ply")

            colors_bak = np.array((colors_bak + 1)/2 * 255, dtype="uint8")
            ply_save = []
            for i in range(len(coords_bak)):
                ply_save.append((coords_bak[i][0], coords_bak[i][1], coords_bak[i][2],
                                 colors_bak[i][0], colors_bak[i][1], colors_bak[i][2], y[i]))
            ply_save = np.array(ply_save,
                                dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                       ("label", "u1")])
            el = PlyElement.describe(ply_save, "vertex")
            plydata = PlyData([el], text=True)
            plydata.write(dst_path_file)

    ret_flop = scn.forward_pass_multiplyAdd_count / 1e6

    print(data_type, ret_data_id, "--- Script time(s): {:.2f}, memory(M): {:.2f}, time(s) {:.2f}, flops(M): {:.2f}"
          .format(time.time() - start_time, ret_memory/n_file, ret_time/n_file, ret_flop/n_file))

    return ret_data_id, ret_time/n_file, ret_flop/n_file, ret_memory/n_file


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
    print("id, time(s), FLOP(M), memory(M)")
    print(np.array_str(result_vstack, precision=2, suppress_small=True))

    save_file = os.path.join(save_dir, "result_memory.csv")
    print("saving file to:", save_file)
    np.savetxt(save_file, result, fmt="%d,%.2f,%.2f,%.2f",
               header="data_id,Time(s),FLOP(M),memory(M)")

