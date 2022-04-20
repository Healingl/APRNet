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

from tqdm import tqdm
if __name__ == "__main__":
    yaml_config = Parser('./config/data_config/iseg2017_cross_validation_settings.yaml')
    print(yaml_config)

    fold_save_csv_dir = os.path.join(yaml_config.csv_dir, 'kfold_%s_data'%(yaml_config.kfold))

    train_all_preproc_data_csv = pd.read_csv(yaml_config.iseg2017_training_preproc_csv_path)

    if not os.path.exists(fold_save_csv_dir):os.makedirs(fold_save_csv_dir)
    fold_list = os.listdir(fold_save_csv_dir)
    print('fold_save_csv_dir',fold_save_csv_dir)
    for fold_name in tqdm(fold_list):

        current_fold_dir = os.path.join(fold_save_csv_dir,fold_name)
        current_fold_train_csv = pd.read_csv(os.path.join(current_fold_dir,'train.csv'))
        current_fold_val_csv = pd.read_csv(os.path.join(current_fold_dir,'val.csv'))

        current_fold_train_patient_id_df = current_fold_train_csv[['patient_id']]
        current_fold_val_patient_id_df = current_fold_val_csv[['patient_id']]

        current_fold_train_preproc_data_csv = pd.merge(train_all_preproc_data_csv,
                                                            current_fold_train_patient_id_df, how='inner',
                                                            on=['patient_id'])
        current_fold_val_preproc_data_csv = pd.merge(train_all_preproc_data_csv,
                                                          current_fold_val_patient_id_df, how='inner', on=['patient_id'])

        current_fold_train_preproc_data_path = os.path.join(current_fold_dir,'train_preproc.csv')
        current_fold_val_preproc_data_path = os.path.join(current_fold_dir,
                                                           'val_preproc.csv')

        current_fold_train_preproc_data_csv.to_csv(current_fold_train_preproc_data_path,index=False)
        current_fold_val_preproc_data_csv.to_csv(current_fold_val_preproc_data_path,index=False)

        print('*' * 60, 'fold: %s' % (fold_name), '*' * 60)
        print('train preproc num:%s, val_preproc num:%s'%(len(current_fold_train_preproc_data_csv),len(current_fold_val_preproc_data_csv)))