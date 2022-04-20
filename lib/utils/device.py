#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: Brain3DMRBrainS13RandomLight
# @IDE: PyCharm
# @File: device.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 21-1-27
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import torch
from collections import OrderedDict


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device

def memory_usage_report(device, logger=None):
    max_memory_allocated = round(float(torch.cuda.max_memory_allocated(device=device))/(10**9),4)
    max_memory_cached = round(float(torch.cuda.max_memory_cached(device=device))/(10**9),4)
    memory_allocated = round(float(torch.cuda.memory_allocated(device=device)) / (10 ** 9),4)
    memory_cached = round(float(torch.cuda.memory_cached(device=device)) / (10 ** 9),4)


    if logger is not None:
        logger.info("Max memory allocated on device {}: ".format(device) + str(max_memory_allocated) + "GB.")
        logger.info("Max memory cached on device {}: ".format(device)+ str(max_memory_cached) + "GB.")
        logger.info("memory allocated on device {}: ".format(device) + str(memory_allocated) + "GB.")
        logger.info("memory cached on device {}: ".format(device) + str(memory_cached) + "GB.")


def dict_conversion(d):
    new_state_dict = OrderedDict()
    for k, v in d.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict