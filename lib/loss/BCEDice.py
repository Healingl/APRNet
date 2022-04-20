#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: BrainTS2020SegDeepSupTorch16
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2021/12/27
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss+Dice
    """

    def __init__(self, do_sigmoid=True):
        super(BCEDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
    def compute_intersection(self, inputs, targets):

        intersection = torch.sum(inputs * targets)

        return intersection

    def metric_dice_compute(self, inputs_logits, targets, smooth = 1e-6):
        input_onehot = inputs_logits > 0.5
        input_onehot = input_onehot.float()

        metric_dice = (2*torch.sum(input_onehot * targets)+smooth) / (input_onehot.sum() + targets.sum() + smooth)

        return metric_dice.data.cpu().numpy()

    def binary_dice_loss(self, inputs, targets, smooth = 1e-6, pow = False):
        metric_dice = self.metric_dice_compute(inputs, targets, smooth)
        intersection = self.compute_intersection(inputs, targets)
        dice_loss = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        if pow:
            dice_loss = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)

        dice_loss = 1 - dice_loss
        return dice_loss, metric_dice


    def bce_loss(self, inputs, targets):
        targets = targets.float()
        bce_criterion = nn.BCEWithLogitsLoss()
        bce_loss = bce_criterion(input=inputs, target=targets)

        return bce_loss

    def forward(self, inputs, target):
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)
            # print(inputs)
            # print(self.nomalization(inputs)[0][0][0])
            # print(">>>>>" * 20)
            # print(torch.sigmoid(inputs)[0][0][0])
            # print(">>>>>"*20)
            # print(F.sigmoid(inputs)[0][0][0])
            # assert False
        b,c,z,y,x = target.size()

        channel_dice_list = []
        dice_loss_sum = 0
        for i in range(c):
            current_channel_dice_loss, current_channel_metric_dice = self.binary_dice_loss(inputs[:, i, ...], target[:, i, ...])
            dice_loss_sum = dice_loss_sum + current_channel_dice_loss
            channel_dice_list.append(current_channel_metric_dice)
        mean_dice_loss = dice_loss_sum / c

        bce_loss = self.bce_loss(inputs=inputs, targets=target)
        combo_loss = 0.5*mean_dice_loss + 0.5*bce_loss
        return combo_loss, channel_dice_list


