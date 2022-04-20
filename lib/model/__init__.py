#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from lib.model.NoNewUNet import NoNewUNet
from lib.model.APRNet import APRNet
from lib.model.UNet3D import UNet3D
def create_model(model_config_args):
    model_name = model_config_args.model_name

    model_list = ['NoNewUNet','APRNet', 'UNet3D']

    assert model_name in model_list

    in_channels = model_config_args.input_channels
    num_classes = model_config_args.num_classes
    input_shape = model_config_args.data_shape

    # print('--' * 50)
    # print('Load Model Name: %s'%(model_name))
    # print('net_params:')
    # print('\t',model_config_args["net_params"])

    # NoNewUNet
    if model_name == 'NoNewUNet':
        model = NoNewUNet(in_channels=in_channels,
                          classes=num_classes,
                          base_n_filter=model_config_args["net_params"]["base_n_filter"],
                          )
    elif model_name == 'APRNet':
        model = APRNet(in_channels=in_channels,
                          classes=num_classes,
                          base_n_filter=model_config_args["net_params"]["base_n_filter"],
                          depth=model_config_args["net_params"]["depth"]
                          )

    elif model_name == 'UNet3D':
        model = UNet3D(in_channels=in_channels, out_channels=num_classes, init_features=model_config_args["net_params"]["base_n_filter"])
    else:
        assert False

    model.count_params()

    print('Success Modle Structure Init!')
    print('--' * 50)
    torch.cuda.empty_cache()
    return model
