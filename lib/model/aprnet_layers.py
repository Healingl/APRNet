#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 22-4-15
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool3D(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate3D(nn.Module):
    def __init__(self):
        super(SpatialGate3D, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class MCDAttention(nn.Module):
    """
    Multi-modal Cross-Dimension Attention Module
    """
    def __init__(self, ):
        super(MCDAttention, self).__init__()
        self.ChannelGateD = SpatialGate3D()
        self.ChannelGateH = SpatialGate3D()
        self.ChannelGateW = SpatialGate3D()
        self.SpatialGate = SpatialGate3D()
    def forward(self, x):
        # batch, c, d, h, w

        # calculate d attention
        x_perm1 = x.permute(0, 2, 1, 3, 4).contiguous()
        x_out1 = self.ChannelGateD(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3, 4).contiguous()

        # calculate h attention
        x_perm2 = x.permute(0, 3, 2, 1, 4).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1, 4).contiguous()

        # calculate w attention
        x_perm3 = x.permute(0, 4, 2, 3, 1).contiguous()
        x_out3 = self.ChannelGateH(x_perm3)
        x_out31 = x_out3.permute(0, 4, 2, 3, 1).contiguous()

        # calculate channel attention
        x_out_spatial = self.SpatialGate(x)

        x_out = (1 / 4) * (x_out_spatial + x_out11 + x_out21 + x_out31)

        return x_out


class MCDAResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCDAResBlock, self).__init__()
        self.mcda_module = MCDAttention()
        self.conv = nn.Conv3d(2*in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                             padding=(0, 0, 0), bias=False)
        self.in3d = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        shortcut = x
        out = self.mcda_module(x)
        out = torch.cat((out, shortcut),dim=1)
        out = self.conv(out)
        out = F.leaky_relu(self.in3d(out), inplace=True)
        return out

# batch, c, d, h, w
class APC3D(nn.Module):
    """
    3D Anisotropic Pyramidal Convolution
    """
    def __init__(self, num_channels,  norm_type=None, act_type=None):
        super(APC3D, self).__init__()
        self.norm_type = norm_type
        self.act_type = act_type
        if norm_type:
            self.norm = norm_type(num_channels)
        if act_type:
            self.act = act_type()

        self.pyramidconv_3x3x3 = nn.Conv3d(num_channels, num_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=((3-1)// 2,(3-1) // 2,(3-1) // 2),bias=False)
        self.pyramidconv_1x5x5 = nn.Conv3d(num_channels, num_channels, kernel_size=(1,5,5), stride=(1,1,1), padding=((1-1)// 2,(5-1) // 2,(5-1) // 2),bias=False)
        # self.pyramidconv_1x7x7 = nn.Conv3d(num_channels, num_channels, kernel_size=(1,7,7), stride=(1,1,1), padding=((1-1)// 2,(7-1) // 2,(7-1) // 2),bias=False)


        self.pyramidconv_3x1x1_2 = nn.Conv3d(num_channels, num_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                           padding=((3 - 1) // 2, (1 - 1) // 2, (1 - 1) // 2), bias=False)
        # self.pyramidconv_3x1x1_3 = nn.Conv3d(num_channels, num_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
        #                                      padding=((3 - 1) // 2, (1 - 1) // 2, (1 - 1) // 2), bias=False)


        self.pyramidconv_1x1x1_2 = nn.Conv3d(2* num_channels, num_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                             padding=(0, 0, 0), bias=False)



    def forward(self, x):
        x_pyramidconv_3x3x3 = self.pyramidconv_3x3x3(x)
        x_pyramidconv_3x5x5 = self.pyramidconv_3x1x1_2(self.pyramidconv_1x5x5(x))
        # x_pyramidconv_3x7x7 =self.pyramidconv_3x1x1_3(self.pyramidconv_1x7x7(x))
        x = self.pyramidconv_1x1x1_2(torch.cat((x_pyramidconv_3x3x3, x_pyramidconv_3x5x5), dim=1))
        if self.norm_type:
            x = self.norm(x)
        if self.act_type:
            x = self.act(x)
        return x


from lib.model.layers.utils import count_param
if __name__ == '__main__':
    feature_num = 16

    net = APC3D(num_channels = feature_num, norm_type=nn.InstanceNorm3d, act_type=nn.LeakyReLU)
    # net = MCDAResBlock(in_channels=feature_num,out_channels=feature_num//2)

    # print(net)

    param = count_param(net)
    print('net totoal parameters: %.5fM (%d)' % (param / 1e6, param))
    net.eval()
    with torch.no_grad():
        input_tensor = torch.rand(1, feature_num, 64, 96, 96)
        out_tensor = net(input_tensor)
        print(out_tensor.shape)

