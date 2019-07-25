import torch
import numpy as np
import pptk
import math




data = torch.load("log/test2.pth")
coords, colors, labels = data

scale = 1

m = np.eye(3)
m *= scale
theta = 0
m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
a = np.matmul(coords, m)


print("Before")
print(a.min(axis=0))
print(a.max(axis=0))
print(a.mean(axis=0))

scale = 20
full_scale = 4096
# pptk.viewer(a)

m = np.eye(3)
# m[0][0] *= np.random.randint(0, 2)*2-1
m *= scale
theta = 0
m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
a = np.matmul(coords, m) + full_scale/2
m = a.min(0)
M = a.max(0)
q = M-m
#offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) * np.random.rand(3) \
#         + np.clip(full_scale - M + m + 0.001, None, 0) * np.random.rand(3)
#print(offset)
# a += offset
# idxs = (a.min(1) >= 0)*(a.max(1) < full_scale)
# a = a[idxs]
print("After")
print(a.min(axis=0))
print(a.max(axis=0))
print(a.mean(axis=0))

