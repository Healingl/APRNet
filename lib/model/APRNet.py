#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 22-4-15
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #


import torch.nn as nn
import torch
from torchsummary import summary
from lib.model.BaseModelClass import BaseModel
from lib.model.aprnet_layers import APC3D, MCDAResBlock
import torch.nn.functional as F
from lib.model.revtorch import revtorch as rv

class ResidualInner(nn.Module):
    def __init__(self, channels):
        super(ResidualInner, self).__init__()
        self.conv = APC3D(num_channels = channels, norm_type=nn.InstanceNorm3d, act_type=nn.LeakyReLU)

    def forward(self, x):
        x = self.conv(x)
        return x

def makeReversibleSequence(channels):
    innerChannels = channels // 2

    fBlock = ResidualInner(innerChannels)
    gBlock = ResidualInner(innerChannels)

    return rv.ReversibleBlock(fBlock, gBlock)

def makeReversibleComponent(channels, blockCount):
    modules = []
    for i in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))

class APRNet(BaseModel):
    """
    Y. Zhuang, H. Liu, E. Song, G. Ma, X. Xu and C. -C. Hung,
    "APRNet: A 3D Anisotropic Pyramidal Reversible Network With Multi-Modal Cross-Dimension Attention for Brain Tissue Segmentation in MR Images,"
    in IEEE Journal of Biomedical and Health Informatics,
    doi: 10.1109/JBHI.2021.3093932.
    """
    def __init__(self, in_channels, classes, base_n_filter=16, depth=2):
        super(APRNet, self).__init__()

        assert in_channels==2
        self.in_channels = in_channels
        self.n_classes = classes
        self.base_n_filter = base_n_filter
        self.depth = depth

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Encoder 1
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        encoder_1_base_n_filter = self.base_n_filter
        # top
        self.encoder_top_1 = nn.Sequential()
        self.encoder_top_1.add_module('encoder_top_1_conv', nn.Conv3d(1, encoder_1_base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False))
        self.encoder_top_1.add_module('encoder_top_1_rs',makeReversibleComponent(encoder_1_base_n_filter, depth))

        # bottom
        self.encoder_bottom_1 = nn.Sequential()
        self.encoder_bottom_1.add_module('encoder_bottom_1_conv',
                                      nn.Conv3d(1, encoder_1_base_n_filter, kernel_size=3, stride=1,
                                                padding=1,
                                                bias=False))
        self.encoder_bottom_1.add_module('encoder_bottom_1_rs',makeReversibleComponent(encoder_1_base_n_filter, depth))


        # # # # # # # # # # # # # # # #
        # Channel Attention
        # # # # # # # # # # # # # # # #
        self.encoder_stage_ca_1 = nn.Sequential()
        self.encoder_stage_ca_1.add_module('stage_channel_attention_1_conv',
                                           MCDAResBlock(2 * encoder_1_base_n_filter,
                                                                     encoder_1_base_n_filter))

        # self.encoder_stage_ca_1.add_module('stage_channel_attention_1_ta',
        #                                     ProjectExciteLayerResBlock(num_channels=encoder_1_base_n_filter))

        # # # # # # # # # # # # # # # #
        # DownSample
        # # # # # # # # # # # # # # # #
        self.encoder_top_1_downsample = nn.Conv3d(encoder_1_base_n_filter, encoder_1_base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_bottom_1_downsample = nn.Conv3d(encoder_1_base_n_filter, encoder_1_base_n_filter * 2, kernel_size=3, stride=2,
                                                  padding=1, bias=False)


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Encoder 2
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        encoder_2_base_n_filter = 2*self.base_n_filter
        # top
        self.encoder_top_2 = nn.Sequential()
        self.encoder_top_2.add_module('encoder_top_2_rs', makeReversibleComponent(encoder_2_base_n_filter, depth))

        # bottom
        self.encoder_bottom_2 = nn.Sequential()
        self.encoder_bottom_2.add_module('encoder_bottom_2_rs', makeReversibleComponent(encoder_2_base_n_filter, depth))



        # # # # # # # # # # # # # # # #
        # Channel Attention
        # # # # # # # # # # # # # # # #
        self.encoder_stage_ca_2 = nn.Sequential()
        self.encoder_stage_ca_2.add_module('stage_channel_attention_2_conv',
                                           MCDAResBlock(2 * encoder_2_base_n_filter,
                                                                     encoder_2_base_n_filter))


        # # # # # # # # # # # # # # # #
        # DownSample
        # # # # # # # # # # # # # # # #
        self.encoder_top_2_downsample = nn.Conv3d(encoder_2_base_n_filter, encoder_2_base_n_filter * 2, kernel_size=3,
                                                  stride=2, padding=1, bias=False)
        self.encoder_bottom_2_downsample = nn.Conv3d(encoder_2_base_n_filter, encoder_2_base_n_filter * 2,
                                                     kernel_size=3, stride=2,
                                                     padding=1, bias=False)


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Gating and Center
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        gating_base_n_filter = 4 * self.base_n_filter
        attention_dsample = (2, 2, 2)

        # top
        self.encoder_center_top = nn.Sequential()
        self.encoder_center_top.add_module('center_top_rs', makeReversibleComponent(gating_base_n_filter,depth))

        # bottom
        self.encoder_center_bottom = nn.Sequential()
        self.encoder_center_bottom.add_module('center_bottom_rs', makeReversibleComponent(gating_base_n_filter,depth))




        # # # # # # # # # # # # # # # #
        # Channel Attention
        # # # # # # # # # # # # # # # #
        self.encoder_stage_ca_3 = nn.Sequential()
        self.encoder_stage_ca_3.add_module('stage_channel_attention_3_conv',
                                           MCDAResBlock(2 * gating_base_n_filter, gating_base_n_filter))
        # # # # # # # # # # # # # # # #
        # UpSample
        # # # # # # # # # # # # # # # #
        # stage
        self.encoder_center_stage_upsample = nn.Sequential()
        self.encoder_center_stage_upsample.add_module('gating_stage_upsample_conv',
                                                    self.upsample_conv(gating_base_n_filter, gating_base_n_filter))
        self.encoder_center_stage_upsample.add_module('gating_stage_upsample',
                                                    self.norm_lrelu_upscale_conv_norm_lrelu(gating_base_n_filter,
                                                                                            encoder_2_base_n_filter))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Decoder 2
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        decoder_2_base_n_filter = 4*self.base_n_filter


        # stage
        self.decoder_2_stage = nn.Sequential()
        self.decoder_2_stage.add_module('decoder_2_stage_rs', makeReversibleComponent(decoder_2_base_n_filter,depth))

        # # # # # # # # # # # # # # # #
        # UpSample
        # # # # # # # # # # # # # # # #
        # stage
        self.decoder_2_stage_upsample = nn.Sequential()
        self.decoder_2_stage_upsample.add_module('decoder_2_stage_upsample_conv',
                                                       self.upsample_conv(decoder_2_base_n_filter, decoder_2_base_n_filter//2))
        self.decoder_2_stage_upsample.add_module('decoder_2_stage_upsample',
                                                       self.norm_lrelu_upscale_conv_norm_lrelu(decoder_2_base_n_filter//2, decoder_2_base_n_filter//4))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Decoder 1
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        decoder_1_base_n_filter = 2 * self.base_n_filter



        # stage
        self.decoder_1_stage = nn.Sequential()
        self.decoder_1_stage.add_module('decoder_1_stage_rs', makeReversibleComponent(decoder_1_base_n_filter,depth))

        ### conv
        # stage
        self.decoder_1_stage_output_conv = nn.Sequential()
        self.decoder_1_stage_output_conv.add_module('decoder_1_stage_output_conv', self.upsample_conv(decoder_1_base_n_filter, decoder_1_base_n_filter//2))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Classifier
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.classifier = nn.Sequential()
        self.classifier.add_module('aggregation',self.conv_norm_lrelu(feat_in=base_n_filter,feat_out=base_n_filter))
        self.classifier.add_module('classifier', nn.Conv3d(base_n_filter,
                                                           classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))
    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU(inplace=True))
    def upsample_conv(self,feat_in, feat_out):

        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU(inplace=True))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU(inplace=True))


    def conv_norm_lrelu_gating(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        in_stream_top = x[:, 0, ...].unsqueeze(dim=1)
        in_stream_bottom = x[:, 1, ...].unsqueeze(dim=1)


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # * * Three Path Encoder
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # #
        # Encoder 1
        # # # # # # # # # # # # # # # #
        in_stream_top = self.encoder_top_1(in_stream_top)
        in_stream_bottom = self.encoder_bottom_1(in_stream_bottom)


        # context
        in_stream_stage_encoder_1_feature_context = self.encoder_stage_ca_1(torch.cat((in_stream_top, in_stream_bottom), dim=1))

        # downsample
        in_stream_top = self.encoder_top_1_downsample(in_stream_top)
        in_stream_bottom = self.encoder_bottom_1_downsample(in_stream_bottom)



        # # # # # # # # # # # # # # # #
        # Encoder 2
        # # # # # # # # # # # # # # # #
        in_stream_top = self.encoder_top_2(in_stream_top)
        in_stream_bottom = self.encoder_bottom_2(in_stream_bottom)



        # context
        in_stream_stage_encoder_2_feature_context = self.encoder_stage_ca_2(torch.cat((in_stream_top, in_stream_bottom), dim=1))

        # downsample
        in_stream_top = self.encoder_top_2_downsample(in_stream_top)
        in_stream_bottom = self.encoder_bottom_2_downsample(in_stream_bottom)

        # # # # # # # # # # # # # # # #
        # Bottom
        # # # # # # # # # # # # # # # #
        # center
        in_stream_top = self.encoder_center_top(in_stream_top)
        in_stream_bottom = self.encoder_center_bottom(in_stream_bottom)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # * * Concat Three Path
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # context
        in_stream_stage_gating_feature_context = self.encoder_stage_ca_3(torch.cat((in_stream_top, in_stream_bottom), dim=1))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # * * One Path Decoder
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # upsample
        in_stream_stage = self.encoder_center_stage_upsample(in_stream_stage_gating_feature_context)

        # # # # # # # # # # # # # # # #
        # Decoder 2
        # # # # # # # # # # # # # # # #
        # stage
        in_stream_stage = torch.cat([in_stream_stage, in_stream_stage_encoder_2_feature_context], dim=1)

        # # TA
        # in_stream_stage = self.decoder_2_stage_ta(in_stream_stage)

        in_stream_stage = self.decoder_2_stage(in_stream_stage)

        # upsample
        in_stream_stage = self.decoder_2_stage_upsample(in_stream_stage)

        # # # # # # # # # # # # # # # #
        # Decoder 1
        # # # # # # # # # # # # # # # #
        # stage
        in_stream_stage = torch.cat([in_stream_stage, in_stream_stage_encoder_1_feature_context], dim=1)


        in_stream_stage = self.decoder_1_stage(in_stream_stage)

        # output
        in_stream_stage = self.decoder_1_stage_output_conv(in_stream_stage)

        seg_layer = self.classifier(in_stream_stage)
        return seg_layer

    def test(self, device='cpu'):

        pass



from lib.model.layers.utils import count_param
if __name__ == '__main__':
    n_modal = 2
    n_classes = 4
    net = APRNet(in_channels=n_modal,classes=n_classes,base_n_filter=48,depth=1)
    param = count_param(net)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))
