#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: Brain3DISEG17
# @IDE: PyCharm
# @File: iseg17_metrics.py
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 20-12-5
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
from lib.eval.binary_metric import dc, hd95, asd

def iseg17_eval_metrics(gt,pred,class_label_list=[1,2,3],show=True):
    """
    0.Background (everything outside the brain)
    1.Cerebrospinal fluid (including ventricles)
    2.Gray matter (cortical gray matter and basal ganglia)
    3.White matter (including white matter lesions)
    :param gt: 3D array
    :param pred: 3D array
    :param class_label_list: [1,2,3]
    :return:
    """
    # hd95(result, reference, voxelspacing=None, connectivity=1):

    label_dict = {'1':'CSF','2':'GM','3':'WM'}

    pred = pred.astype(dtype='int')
    gt=gt.astype(dtype='int')

    dsc_dict = {}
    hd95_dict = {}
    asd_dict = {}
    for current_label in class_label_list:
        current_label = int(current_label)

        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==current_label)]=1
        y_c[np.where(pred==current_label)]=1

        try:
            # result, reference
            current_label_dsc = dc(y_c,gt_c)
        except:
            print('dc error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_dsc = 0
        try:
            current_label_hd95 = hd95(y_c,gt_c,voxelspacing=(1,1,1))
        except:
            print('hd95 error gt:max %s, min %s, y_c:max %s, min %s' % (gt_c.max(), gt_c.min(), y_c.max(), y_c.min()))
            current_label_hd95 = 0
        try:
            current_label_asd = asd(y_c,gt_c,voxelspacing=(1,1,1))
        except:
            print('asd error gt:max %s, min %s, y_c:max %s, min %s'%(gt_c.max(),gt_c.min(),y_c.max(),y_c.min()))
            current_label_asd = 0

        dsc_dict['%s' % (label_dict[str(current_label)])] = round(current_label_dsc,4)
        hd95_dict['%s' % (label_dict[str(current_label)])] = round(current_label_hd95,4)
        asd_dict['%s' % (label_dict[str(current_label)])] = round(current_label_asd, 4)
    if show:
        print('>>>'*30)
        print('DSC:',dsc_dict)
        print('HD95:',hd95_dict)
        print('ASD:',asd_dict)
        print('>>>'*30)
    return dsc_dict,hd95_dict,asd_dict