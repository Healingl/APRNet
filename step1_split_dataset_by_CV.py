#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
from sklearn.model_selection import KFold
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from ast import literal_eval
import copy
import pandas as pd
from lib.utils.simple_parser import Parser



if __name__ == "__main__":
    yaml_config = Parser('./config/data_config/iseg2017_cross_validation_settings.yaml')
    save_csv_dir = yaml_config['csv_dir']
    train_all_data_csv_path = yaml_config['iseg2017_training_raw_csv_path']

    if not os.path.exists(save_csv_dir): os.makedirs(save_csv_dir)

    train_data_dir = yaml_config['raw_train_data_dir']
    if not os.path.exists(train_data_dir): assert False

    # "patient_id", "t1", "t2", "seg"
    train_data_all_patient_id_list = [i for i in range(1,11)]

    train_all_data_list = []
    for idx, current_patient_id in enumerate(train_data_all_patient_id_list):
        subject_name = 'subject-%d' % current_patient_id
        current_patient_t1_path = os.path.abspath(os.path.join(train_data_dir, subject_name+'-T1.hdr'))
        current_patient_t2_path = os.path.abspath(os.path.join(train_data_dir, subject_name+'-T2.hdr'))
        current_patient_seg_path = os.path.abspath(os.path.join(train_data_dir, subject_name+'-label.hdr'))

        if not os.path.exists(current_patient_t1_path) or not os.path.exists(current_patient_t2_path) or not os.path.exists(current_patient_seg_path):
            print(current_patient_id)
            assert False

        train_all_data_list.append(
            [subject_name, current_patient_t1_path, current_patient_t2_path, current_patient_seg_path])

    all_patient_data_csv = pd.DataFrame(
        columns=["patient_id", "t1", "t2", "seg"], data=train_all_data_list)
    all_patient_data_csv.to_csv(train_all_data_csv_path, index=False)

    # cross validation by k-fold
    kfold = yaml_config['kfold']
    random_state = yaml_config['random_seed']

    kfold_save_csv_dir = os.path.join(save_csv_dir, 'kfold_%s_data' % (kfold))
    if not os.path.exists(kfold_save_csv_dir): os.makedirs(kfold_save_csv_dir)
    kf = KFold(n_splits=int(kfold), shuffle=True, random_state=random_state)
    for fold_, (train_index, valid_index) in enumerate(kf.split(all_patient_data_csv)):
        print('*' * 30, 'current fold %s' % (fold_), '*' * 30)
        this_fold_save_csv_dir = os.path.join(kfold_save_csv_dir, '%s' % (fold_))
        if not os.path.exists(this_fold_save_csv_dir): os.makedirs(this_fold_save_csv_dir)
        #
        this_fold_train_data_csv = all_patient_data_csv.iloc[train_index]
        this_fold_val_data_csv = all_patient_data_csv.iloc[valid_index]
        print('all num: %s, train num: %s, val num: %s' % (
        len(all_patient_data_csv), len(this_fold_train_data_csv), len(this_fold_val_data_csv)))
        print('train', train_index)
        print('val', valid_index)
        this_fold_train_data_csv.to_csv(os.path.join(this_fold_save_csv_dir, 'train.csv'), index=False)
        this_fold_val_data_csv.to_csv(os.path.join(this_fold_save_csv_dir, 'val.csv'), index=False)




