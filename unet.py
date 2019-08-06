# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import data
import iou

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np

# Options
m = 16  # 16 or 32; 16
residual_blocks = False  # True or False; False
block_reps = 2  # Conv block repetition factor: 1 or 2; 1

# number of parameters
# m, block_reps, residual_blocks
# 16, 1, False: 2689860
# 16, 1, True:  4334692
# 16, 2, False: 4288100
# 16, 2, True:  7531172 !!!
# 32, 1, False: 10748532
# 32, 1, True:  17324724
# 32, 2, False: 17138356
# 32, 2, True:  30104372

use_cuda = torch.cuda.is_available()
# exp_name = 'unet_scale20_m16_rep1_notResidualBlocks'
exp_name = 'log/scannet_m{}_rep{}_residual{}'.format(m, block_reps, residual_blocks)


class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(data.dimension, data.full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(data.dimension, 3, m, 3, False)).add(
               scn.UNet(data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(data.dimension))
        self.linear = nn.Linear(m, 20)

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x


unet = Model()
if use_cuda:
    unet = unet.cuda()

training_epochs = 1024
training_epoch = scn.checkpoint_restore(unet, exp_name, 'unet', use_cuda)
optimizer = optim.Adam(unet.parameters())
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))


def my_checkpoint_save(model, exp_name, name2, epoch, use_cuda=True):
    f = exp_name+'-%09d' % epoch + '.pth'
    model.cpu()
    torch.save(model.state_dict(), f)
    if use_cuda:
        model.cuda()


for epoch in range(training_epoch, training_epochs+1):
    unet.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count = 0
    scn.forward_pass_hidden_states = 0
    start = time.time()
    train_loss = 0
    for i, batch in enumerate(data.train_data_loader):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()
            batch['y'] = batch['y'].cuda()
        predictions = unet(batch['x'])
        loss = torch.nn.functional.cross_entropy(predictions, batch['y'])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(epoch, 'Train loss', train_loss/(i+1), 'MegaMulAdd=', scn.forward_pass_multiplyAdd_count/len(data.train)/1e6,
          'MegaHidden', scn.forward_pass_hidden_states/len(data.train)/1e6, 'time=', time.time() - start, 's')
    scn.checkpoint_save(unet, exp_name, 'unet', epoch, use_cuda)

    if epoch % 10 == 0:
        my_checkpoint_save(unet, exp_name, 'unet', epoch, use_cuda)
        with torch.no_grad():
            unet.eval()
            store = torch.zeros(data.valOffsets[-1], 20)
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0
            start = time.time()
            for rep in range(1, 1+data.val_reps):
                for i, batch in enumerate(data.val_data_loader):
                    if use_cuda:
                        batch['x'][1] = batch['x'][1].cuda()
                        batch['y'] = batch['y'].cuda()
                    predictions = unet(batch['x'])
                    store.index_add_(0, batch['point_ids'], predictions.cpu())
                print(epoch, rep, 'Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count/len(data.val)/1e6, 'MegaHidden',
                      scn.forward_pass_hidden_states/len(data.val)/1e6, 'time=', time.time() - start, 's')
                iou.evaluate(store.max(1)[1].numpy(), data.valLabels)
