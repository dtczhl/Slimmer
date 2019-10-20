"""
    add label to missing points
"""

import numpy as np
import glob
import os
import pandas as pd

scannet_dir = "/home/dtc/Backup/Data/ScanNet"

# ------ Configuration ------

# Random, Grid, Hierarchy
data_type = "Random"

specify_id = 40

# --- end of configuration ---

original_ply_dir = os.path.join(scannet_dir, "Ply")
simplified_pth_dir = os.path.join(scannet_dir, "PlyLabel", data_type, str(specify_id))

save_dir = os.path.join(scannet_dir, "PlyLabelAdd", data_type, str(specify_id))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ply_base_files = glob.glob(os.path.join(original_ply_dir, "*.ply"))
for ply_base_file in ply_base_files:
    ply_base_name = os.path.basename(ply_base_file)

    orig_file = ply_base_file
    pred_file = os.path.join(simplified_pth_dir, ply_base_name)
    save_file = os.path.join(save_dir, ply_base_name[:-4]+".txt")

    mycmd = "../Cpp/add_label/build/add_label {} {} {}"\
            .format(orig_file, pred_file, save_file)
    os.system(mycmd)




