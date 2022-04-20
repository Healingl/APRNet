#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 22-4-20
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

import torch
import numpy as np
import math
import torch.nn.functional as F

def get_order_value_list(start_idx,input_volume_axis, crop_shape_axi, extraction_step_axi):
    start_idx_list = [start_idx]
    while start_idx < input_volume_axis - crop_shape_axi:
        start_idx += extraction_step_axi
        if start_idx > input_volume_axis - crop_shape_axi:
            start_idx = input_volume_axis - crop_shape_axi
            start_idx_list.append(start_idx)
            break
        start_idx_list.append(start_idx)
    return start_idx_list

def get_order_crop_list(volume_shape,crop_shape,extraction_step):
    """
    :param volume_shape: e.g.(155,240,240)
    :param crop_shape: e.g.(128,128,128)
    :param extraction_step: e.g.(128,128,128)
    :return:
    """
    assert volume_shape[0] >= crop_shape[0], "crop size is too big"
    assert volume_shape[1] >= crop_shape[1], "crop size is too big"
    assert volume_shape[2] >= crop_shape[2], "crop size is too big"
    crop_z_list = get_order_value_list(start_idx=0,input_volume_axis=volume_shape[0],crop_shape_axi=crop_shape[0],extraction_step_axi=extraction_step[0])
    crop_y_list = get_order_value_list(start_idx=0,input_volume_axis=volume_shape[1],crop_shape_axi=crop_shape[1],extraction_step_axi=extraction_step[1])
    crop_x_list =get_order_value_list(start_idx=0,input_volume_axis=volume_shape[2],crop_shape_axi=crop_shape[2],extraction_step_axi=extraction_step[2])
    crop_list = []
    for current_crop_z_value in crop_z_list:
        for current_crop_y_value in crop_y_list:
            for current_crop_x_value in crop_x_list:
                crop_list.append((current_crop_z_value,current_crop_y_value,current_crop_x_value))
    return crop_list



