import os
import urllib.request
from shutil import copyfile

# ----- Configurations -----

data_dir = "/home/dtczhl/Data/ScanNet"
save_dir = "/home/dtczhl/MyGit/dtc-sparseconvnet"
# --- end of Configurations ---

scannet_url = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/"
train_url = scannet_url + "scannetv2_train.txt"
valid_url = scannet_url + "scannetv2_val.txt"

data_train_txt = os.path.join(data_dir, "scannetv2_train.txt")
data_valid_txt = os.path.join(data_dir, "scannetv2_val.txt")

save_train_dir = os.path.join(save_dir, "train")
save_valid_dir = os.path.join(save_dir, "val")

# download labels
urllib.request.urlretrieve(train_url, data_train_txt)
urllib.request.urlretrieve(valid_url, data_valid_txt)

with open(data_train_txt, "r") as f_in:
    lines = f_in.readlines()
    for line in lines:
        line = line.strip()
        f_name = line + "_vh_clean_2.ply"
        f_src = os.path.join(data_dir, "scans/" + line + "/" + f_name)
        f_dst = os.path.join(save_train_dir, f_name)
        copyfile(f_src, f_dst)
        f_name = line + "_vh_clean_2.labels.ply"
        f_src = os.path.join(data_dir, "scans/" + line + "/" + f_name)
        f_dst = os.path.join(save_train_dir, f_name)
        copyfile(f_src, f_dst)

with open(data_valid_txt, "r") as f_in:
    lines = f_in.readlines()
    for line in lines:
        line = line.strip()
        f_name = line + "_vh_clean_2.ply"
        f_src = os.path.join(data_dir, "scans/" + line + "/" + f_name)
        f_dst = os.path.join(save_valid_dir, f_name)
        copyfile(f_src, f_dst)
        f_name = line + "_vh_clean_2.labels.ply"
        f_src = os.path.join(data_dir, "scans/" + line + "/" + f_name)
        f_dst = os.path.join(save_valid_dir, f_name)
        copyfile(f_src, f_dst)
