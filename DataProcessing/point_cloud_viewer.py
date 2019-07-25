"""
    Data Analysis
"""

# x, y, z
#   min: -9.03 -9.63 -3.37
#   max:

import numpy as np
import os
import pptk
import glob

data_dir = "/home/dtc/Data/ScanNet/Bin/"

value_range_min = []
value_range_max = []
for file in glob.glob(os.path.join(data_dir, "*.bin")):
    print(file)
    data = np.fromfile(file, "<f4")
    data = np.reshape(data, (-1, 7))
    value_range_min.append(data.min(axis=0))
    value_range_max.append(data.max(axis=0))

value_range_min = np.array(value_range_min)
value_range_max = np.array(value_range_max)
print(value_range_min.min(axis=0))
print(value_range_max.max(axis=0))
