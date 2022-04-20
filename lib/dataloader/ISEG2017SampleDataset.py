#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
import lib.augment as augment3D
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import rotate


def rotation_zoom3D(input_img_array_list, input_label_array_list, rotation_value=30):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    angle_buffer = np.random.randint(-rotation_value, rotation_value)
    rotate_axes = (1, 2)
    random_rotate_img_array_list = []
    for array_idx, current_np_array in enumerate(input_img_array_list):
        current_rotate_array = rotate(current_np_array, angle_buffer, axes=rotate_axes, reshape=False,
                                      order=0, mode='constant', cval=0)
        random_rotate_img_array_list.append(current_rotate_array)

    random_rotate_label_array_list = []
    for array_idx, current_np_array in enumerate(input_label_array_list):
        current_rotate_array = rotate(current_np_array, angle_buffer, axes=rotate_axes, reshape=False,
                                      order=0, mode='constant', cval=0)
        random_rotate_label_array_list.append(current_rotate_array)

    return random_rotate_img_array_list, random_rotate_label_array_list


class ISEG2017SampleDataset(Dataset):
    def __len__(self):
        return len(self.brain_tissue_csv)

    def __init__(self,
                 csv_file_path,
                 crop_size,
                 mode='train',
                 data_num=-1,
                 use_aug=False,
                 ):
        """
        :param brain_tumor_csv_file_path:
        :param crop_size: (z,y,x)
        :param mode:
        :param data_num:
        :param normalization:
        :param use_aug:
        :param tumor_region:
        :param crop_type:
        """

        assert mode in ['train', 'val'], "error model!"

        # cate_to_label_dict: {'bg':0, "CSF": 1,  'GM': 2, 'WM': 3}

        assert len(crop_size) == 3

        self.mode = mode
        self.use_aug = use_aug
        self.crop_size = crop_size

        brain_tissue_csv = pd.read_csv(csv_file_path)
        print(">>" * 30, "read brain_tissue_csv:", csv_file_path, 'data num: ', len(brain_tissue_csv),
              ">>" * 30)

        self.brain_tissue_csv = brain_tissue_csv


        if self.use_aug and mode == 'train':
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.RandomFlip(),
                            augment3D.RandomRotation(angle_list=[90, 180, 270], axis=0),
                            # augment3D.RandomIntensityScale(),
                            # augment3D.RandomIntensityShift(),
                            # augment3D.GaussianNoise()
                            ],
                p=0.5)

        if data_num == -1:
            self.brain_tissue_csv = self.brain_tissue_csv
        else:
            self.brain_tissue_csv = self.brain_tissue_csv.head(data_num)

        print(">>" * 30, "load data num :", len(self.brain_tissue_csv), ">>" * 30)

    def __getitem__(self, index):
        """
        patient_id,t1ce,t2,flair,LBSA,EA,NA,CA
        """

        # patient_id,whole_brain_size,t1,t2,seg
        current_select_row = self.brain_tissue_csv.iloc[index]

        current_patient_id = current_select_row['patient_id']
        # print('current_patient_id',current_patient_id)
        current_t1_path = current_select_row['t1']
        current_t2_path = current_select_row['t2']

        # all regions
        current_seg_all_regions_path = current_select_row['seg']

        # read array
        current_brain_t1_array = np.load(current_t1_path)
        current_brain_t2_array = np.load(current_t2_path)
        current_brain_seg_gt_array = np.load(current_seg_all_regions_path)

        assert current_brain_t1_array.shape == current_brain_t2_array.shape == current_brain_seg_gt_array.shape

        full_vol_dim = current_brain_t1_array.shape

        # print('full_vol_dim',full_vol_dim)
        assert full_vol_dim[0] >= self.crop_size[0], "source crop size z is too big"
        assert full_vol_dim[1] >= self.crop_size[1], "source crop size y is too big"
        assert full_vol_dim[2] >= self.crop_size[2], "source crop size x is too big"

        # random
        if self.mode == 'train':


            current_random_crop_z = np.random.randint(0, full_vol_dim[0] - self.crop_size[0] + 1)
            current_random_crop_y = np.random.randint(0, full_vol_dim[1] - self.crop_size[1] + 1)
            current_random_crop_x = np.random.randint(0, full_vol_dim[2] - self.crop_size[2] + 1)


        elif self.mode == 'val':
            current_random_crop_z = (full_vol_dim[0] - self.crop_size[0] + 1) // 2
            current_random_crop_y = (full_vol_dim[1] - self.crop_size[1] + 1) // 2
            current_random_crop_x = (full_vol_dim[2] - self.crop_size[2] + 1) // 2

        else:
            assert False
        current_random_crop = (current_random_crop_z, current_random_crop_y, current_random_crop_x)

        # crop
        current_crop_t1_array = self.crop_cube_from_volumn(
            origin_volumn=current_brain_t1_array,
            crop_point=current_random_crop,
            crop_size=self.crop_size
        )

        current_crop_t2_array = self.crop_cube_from_volumn(
            origin_volumn=current_brain_t2_array,
            crop_point=current_random_crop,
            crop_size=self.crop_size
        )


        # current_crop_seg_gt_array
        current_crop_seg_gt_array = self.crop_cube_from_volumn(
            origin_volumn=current_brain_seg_gt_array,
            crop_point=current_random_crop,
            crop_size=self.crop_size
        )

        assert current_crop_seg_gt_array.shape == current_crop_t1_array.shape == current_crop_t2_array.shape ==(self.crop_size[0], self.crop_size[1], self.crop_size[2])



        if self.mode == 'train':
            if self.use_aug:
                [current_crop_t1_array, current_crop_t2_array], current_crop_seg_gt_array = self.transform(
                    [current_crop_t1_array, current_crop_t2_array], current_crop_seg_gt_array)

        # {'bg':0, "CSF": 1,  'GM': 2, 'WM': 3}
        current_crop_seg_bg_array = np.zeros_like(current_crop_seg_gt_array)
        current_crop_seg_bg_array[current_crop_seg_gt_array == 0] = 1

        current_crop_seg_CSF_array = np.zeros_like(current_crop_seg_gt_array)
        current_crop_seg_CSF_array[current_crop_seg_gt_array == 1] = 1

        current_crop_seg_GM_array = np.zeros_like(current_crop_seg_gt_array)
        current_crop_seg_GM_array[current_crop_seg_gt_array == 2] = 1

        current_crop_seg_WM_array = np.zeros_like(current_crop_seg_gt_array)
        current_crop_seg_WM_array[current_crop_seg_gt_array == 3] = 1




        feature_np_array = np.array([current_crop_t1_array, current_crop_t2_array])
        current_crop_seg_gt_array = np.array([current_crop_seg_bg_array, current_crop_seg_CSF_array,
                                              current_crop_seg_GM_array, current_crop_seg_WM_array])

        feature_tensor = torch.from_numpy(feature_np_array).float()
        seg_gt_tensor = torch.from_numpy(current_crop_seg_gt_array).long()

        return feature_tensor, seg_gt_tensor

    def crop_cube_from_volumn(self, origin_volumn, crop_point, crop_size):
        """

        :param origin_volumn: (z,y,x)
        :param crop_point: (current_crop_z, current_crop_y, current_crop_x )
        :param crop_size: (crop_z,crop_y,crop_x)
        :return:
        """
        current_crop_z, current_crop_y, current_crop_x = crop_point

        cube = origin_volumn[current_crop_z:current_crop_z + crop_size[0], current_crop_y:current_crop_y + crop_size[1],
               current_crop_x:current_crop_x + crop_size[2]]

        return cube


from tqdm import tqdm
from torch.utils.data import DataLoader

if __name__ == "__main__":
    iseg2017_train_csv_path = "../../csv/kfold_10_data/0/train_preproc.csv"
    iseg2017_val_csv_path = "../../csv/kfold_10_data/0/val_preproc.csv"

    # brats2018: 155,240,240

    # LBSA,EA,NA,CA
    train_dataset = ISEG2017SampleDataset(
        csv_file_path=iseg2017_train_csv_path,
        crop_size=[64,64,64],
        mode='train',
        data_num=-1,
        use_aug=True,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=0, shuffle=False)

    for i, (feature_tensor, seg_gt_tensor) in tqdm(enumerate(train_loader), total=len(train_loader)):

        print(feature_tensor.shape, seg_gt_tensor.shape)
