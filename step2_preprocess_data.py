#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
import pandas as pd
import numpy as np
from lib.utils.simple_parser import Parser
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk

def convert_iseg2017_label_to_int(origin_seg_array):
    # Converts 0:background / 10:CSF / 150:GM / 250:WM to 0/1/2/3.
    target_seg_array = np.zeros_like(origin_seg_array)
    target_seg_array[origin_seg_array == 10] = 1
    target_seg_array[origin_seg_array == 150] = 2
    target_seg_array[origin_seg_array == 250] = 3
    return target_seg_array


def get_ND_bounding_box(label, margin=0):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if (type(margin) is int):
        margin = [margin] * len(input_shape)
    assert (len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert (dim >= 2 and dim <= 5)
    if (dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif (dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif (dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif (dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if (dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif (dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif (dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out


def read_nii_as_narray(nii_file_path):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # simple itk read
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # read itk image
    patient_itk_data = sitk.ReadImage(nii_file_path)

    #
    origin = np.array(patient_itk_data.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(patient_itk_data.GetSpacing())  # spacing of voxels in world coor. (mm)
    direction = np.array(patient_itk_data.GetDirection())

    # get np array
    patient_volume_narray = sitk.GetArrayFromImage(patient_itk_data)  # z, y, x

    return patient_volume_narray

if __name__ =='__main__':
    yaml_config = Parser('./config/data_config/iseg2017_cross_validation_settings.yaml')
    proproc_data_dir = yaml_config['iseg2017_training_preproc_data_dir']
    training_all_data_csv_path = yaml_config['iseg2017_training_raw_csv_path']

    training_all_preproc_data_csv_path = yaml_config['iseg2017_training_preproc_csv_path']

    if not os.path.exists(proproc_data_dir):os.makedirs(proproc_data_dir)

    train_all_data_csv = pd.read_csv(training_all_data_csv_path)

    # "patient_id", "whole_brain_size", "t1", "t2", "seg"
    preproc_data_list = []
    # # "patient_id", "t1", "t2", "seg"
    for idx, row in tqdm(train_all_data_csv.iterrows(), total=len(train_all_data_csv)):
        # if idx > 1:
        #     break

        current_patient_id = str(row['patient_id'])

        # raw data path
        current_patient_t1_path = row['t1']
        current_patient_t2_path = row['t2']
        current_patient_seg_path = row['seg']

        print("*" * 30, 'all: %s, preproc: %s,  patient_id: %s'%(len(train_all_data_csv), idx, current_patient_id), "*"*30)

        # current patient preproc dir
        current_preproc_volume_save_dir = os.path.join(proproc_data_dir, current_patient_id)
        if not os.path.exists(current_preproc_volume_save_dir):os.makedirs(current_preproc_volume_save_dir)

        # read numpy array
        current_patient_t1_narray = read_nii_as_narray(current_patient_t1_path)
        current_patient_t2_narray = read_nii_as_narray(current_patient_t2_path)
        current_patient_seg_narray = read_nii_as_narray(current_patient_seg_path)

        # convert label
        current_patient_fix_seg_narray = convert_iseg2017_label_to_int(current_patient_seg_narray.astype(np.uint8))

        # whole brain mask
        whole_brain_mask = np.zeros_like(current_patient_t1_narray)
        whole_brain_mask[current_patient_t1_narray > 0] = 1

        # Normalization
        norm_type = 'max_min'
        # 144 192 256
        # mean_std
        if norm_type == 'mean_std':
            inputs_T1_norm = (current_patient_t1_narray - current_patient_t1_narray[whole_brain_mask].mean()) / current_patient_t1_narray[whole_brain_mask].std()
            inputs_T2_norm = (current_patient_t2_narray - current_patient_t2_narray[whole_brain_mask].mean()) / current_patient_t2_narray[whole_brain_mask].std()
        elif norm_type == 'max_min':

            inputs_T1_norm = (current_patient_t1_narray - current_patient_t1_narray.min()) / (current_patient_t1_narray.max()-current_patient_t1_narray.min())
            inputs_T2_norm = (current_patient_t2_narray - current_patient_t2_narray.min()) / (current_patient_t2_narray.max()-current_patient_t2_narray.min())
        else:
            assert False

        # using smallest bounding box to crop whole brain region
        margin = int(64 / 2)
        # smallest bounding box
        bbmin, bbmax = get_ND_bounding_box(whole_brain_mask, margin=margin)

        current_patient_t1_preproc_narray = crop_ND_volume_with_bounding_box(inputs_T1_norm, bbmin, bbmax)
        current_patient_t2_preproc_narray = crop_ND_volume_with_bounding_box(inputs_T2_norm, bbmin, bbmax)
        current_patient_seg_preproc_narray = crop_ND_volume_with_bounding_box(current_patient_fix_seg_narray, bbmin, bbmax)

        assert current_patient_t1_preproc_narray.shape == current_patient_t2_preproc_narray.shape == current_patient_seg_preproc_narray.shape

        current_whole_brain_size =current_patient_t1_preproc_narray.shape
        print('current_whole_brain_size', current_whole_brain_size)


        current_t1_npy_path = os.path.abspath(os.path.join(current_preproc_volume_save_dir, '%s_preproc_t1.npy' % (current_patient_id)))
        current_t2_npy_path = os.path.abspath(os.path.join(current_preproc_volume_save_dir, '%s_preproc_t2.npy' % (current_patient_id)))
        current_seg_npy_path = os.path.abspath(os.path.join(current_preproc_volume_save_dir, '%s_preproc_seg.npy' % (current_patient_id)))
        np.save(current_t1_npy_path, current_patient_t1_preproc_narray)
        np.save(current_t2_npy_path, current_patient_t2_preproc_narray)
        np.save(current_seg_npy_path, current_patient_seg_preproc_narray)


        # "patient_id", "whole_brain_size", "t1", "t2", "seg"
        preproc_data_list.append([str(current_patient_id),
                                  str(current_whole_brain_size),
                                  current_t1_npy_path,
                                  current_t2_npy_path,
                                  current_seg_npy_path])

    preproc_data_csv = pd.DataFrame(columns=["patient_id", "whole_brain_size", "t1", "t2", "seg"],data=preproc_data_list)
    preproc_data_csv.to_csv(training_all_preproc_data_csv_path, index=False)



