#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import torch
from torch.backends import cudnn
cudnn.deterministic = True
cudnn.benchmark = True
cudnn.enabled = True
from torch.utils.data import DataLoader

from lib.model import create_model
from lib.loss import create_loss
from lib.dataloader.ISEG2017SampleDataset import ISEG2017SampleDataset
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from lib.utils.logging import *
from tqdm import tqdm
from lib.utils.simple_parser import Parser

import pandas as pd
import numpy as np

import shutil
import argparse
import importlib

from lib.dataloader import medical_loader_utils as img_utils
from lib.eval.iseg17_metrics import iseg17_eval_metrics
import time
current_time = str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))


def crop_cube_from_volumn(origin_volumn, crop_point, crop_size):
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


def validate_by_patient(model):
    origin_val_csv = pd.read_csv(model_yaml_config.val_data_csv_path)
    val_crop_size = model_yaml_config.crop_size
    val_step_size = model_yaml_config.val_slide_step_size
    val_batch_size = model_yaml_config.val_batch_size
    model.eval()
    with torch.no_grad():
        # patient_id, DSC_CSF, DSC_GM, DSC_WM, HD95_CSF, HD95_GM, HD95_WM, ASD_CSF, ASD_GM, ASD_WM
        iseg2017_eval_results = []
        for idx, row in tqdm(origin_val_csv.iterrows(), total=len(origin_val_csv)):
            # if idx > 1:
            #     break
            current_patient_id = row['patient_id']
            current_t1_path = row['t1']
            current_t2_path = row['t2']
            current_seg_path = row['seg']
            print('>>>>' * 30,
                  'Eval All:%s, Current:%s, Patient id: %s' % (len(origin_val_csv), idx, current_patient_id),
                  '>>>>' * 30)


            origin_t1_array = np.load(current_t1_path)
            origin_t2_array = np.load(current_t2_path)
            origin_seg_array = np.load(current_seg_path)



            full_vol_dim = origin_seg_array.shape

            # sliding windows
            sample_crop_list = img_utils.get_order_crop_list(volume_shape=full_vol_dim,
                                                             crop_shape=val_crop_size,
                                                             extraction_step=val_step_size)

            # (4,155,240,240)
            # prob array
            full_prob_np_array = np.zeros((model_yaml_config.num_classes,
                                           full_vol_dim[0],
                                           full_vol_dim[1],
                                           full_vol_dim[2]))
            # count array
            full_count_np_array = np.zeros((model_yaml_config.num_classes,
                                            full_vol_dim[0],
                                           full_vol_dim[1],
                                           full_vol_dim[2]))

            # batch_size
            PathNum = 0
            temp_crop_list = []
            temp_tensor_list = []

            for current_sample_idx, sample_crop in tqdm(enumerate(sample_crop_list),total=len(sample_crop_list),ncols=50):

                (current_crop_z_value, current_crop_y_value, current_crop_x_value) = sample_crop
                (crop_z_size, crop_y_size, crop_x_size) = val_crop_size



                current_crop_t1_array = crop_cube_from_volumn(
                    origin_volumn=origin_t1_array,
                    crop_point=sample_crop,
                    crop_size=val_crop_size
                )

                current_crop_t2_array = crop_cube_from_volumn(
                    origin_volumn=origin_t2_array,
                    crop_point=sample_crop,
                    crop_size=val_crop_size
                )

                feature_np_array = np.array([current_crop_t1_array, current_crop_t2_array])
                feature_tensor = torch.unsqueeze(torch.from_numpy(feature_np_array).float(),dim=0)

                input_tensor = feature_tensor.cuda(non_blocking=True)

                # 为batch_size预测准备
                PathNum += 1
                temp_crop_list.append(sample_crop)
                temp_tensor_list.append(input_tensor)

                if PathNum == val_batch_size or current_sample_idx == len(sample_crop_list) - 1:
                    input_batch_tensor = torch.cat(temp_tensor_list, dim=0)
                    inputs = input_batch_tensor.cuda(non_blocking=True)
                    del input_batch_tensor

                    outputs = model(inputs)
                    del inputs
                    # 转化成numpy
                    outputs_np = outputs.data.cpu().numpy()

                    for temp_crop_idx in range(len(temp_crop_list)):
                        temp_crop_z_value, temp_crop_y_value, temp_crop_x_value = temp_crop_list[temp_crop_idx]

                        # 获得小块, [4,64,64,64]
                        current_crop_prob_cube = outputs_np[temp_crop_idx]

                        assert len(current_crop_prob_cube) == model_yaml_config.num_classes

                        full_prob_np_array[:, temp_crop_z_value:temp_crop_z_value + crop_z_size,
                        temp_crop_y_value:temp_crop_y_value + crop_y_size,
                        temp_crop_x_value:temp_crop_x_value + crop_x_size] += current_crop_prob_cube[:, :, :, :]
                        full_count_np_array[:, temp_crop_z_value:temp_crop_z_value + crop_z_size,
                        temp_crop_y_value:temp_crop_y_value + crop_y_size,
                        temp_crop_x_value:temp_crop_x_value + crop_x_size] += 1

                    # 清空batch size
                    PathNum = 0
                    temp_crop_list = []
                    temp_tensor_list = []
                    torch.cuda.empty_cache()

                torch.cuda.empty_cache()

            # avoid no overlap region
            full_count_np_array[full_count_np_array == 0] = 1
            predict_seg_array = full_prob_np_array / full_count_np_array

            # (4,155,240,240) -> (155,240,240)
            predict_seg_array = np.argmax(predict_seg_array, axis=0)

            fix_segmentation_map = origin_seg_array
            dsc_dict, hd95_dict, ASD_dict = iseg17_eval_metrics(gt=fix_segmentation_map, pred=predict_seg_array,
                                                                    class_label_list=[1, 2, 3])

            # patient_id, DSC_CSF, DSC_GM, DSC_WM, HD95_CSF, HD95_GM, HD95_WM, ASD_CSF, ASD_GM, ASD_WM
            iseg2017_eval_results.append([current_patient_id,
                                            dsc_dict['CSF'], dsc_dict['GM'], dsc_dict['WM'],
                                            hd95_dict['CSF'], hd95_dict['GM'], hd95_dict['WM'],
                                            ASD_dict['CSF'], ASD_dict['GM'], ASD_dict['WM']
                                            ])

        iseg2017_eval_result_csv = pd.DataFrame(columns=['patient_id', 'DSC_CSF', 'DSC_GM', 'DSC_WM',
                                                           'HD95_CSF', 'HD95_GM', 'HD95_WM',
                                                           'ASD_CSF', 'ASD_GM', 'ASD_WM'], data=iseg2017_eval_results)


        eval_result_str = 'Eval Result: '
        for column_name in iseg2017_eval_result_csv.columns.tolist():
            if column_name != 'patient_id':
                eval_result_str += '%s: %s, '%(column_name, round(iseg2017_eval_result_csv[column_name].mean(),4))
        logger.info('###'*30)
        logger.info(eval_result_str)
        logger.info('###' * 30)

        dice_csf = round(iseg2017_eval_result_csv['DSC_CSF'].mean(),4)
        dice_gm = round(iseg2017_eval_result_csv['DSC_GM'].mean(),4)
        dice_wm = round(iseg2017_eval_result_csv['DSC_WM'].mean(),4)
        mean_dice = round((dice_csf+dice_gm+dice_wm)/3,4)

        return dice_csf,dice_gm,dice_wm,mean_dice

def reproducibility(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    # 输入变量为model config路径
    parser = argparse.ArgumentParser()
    default_model_config_path = './config/model_config/Fold0/UNet3D_Fold0.yaml'
    parser.add_argument("--modelConfigPath", default=default_model_config_path)

    arguments = parser.parse_args()
    configPath = arguments.modelConfigPath
    model_yaml_config = Parser(configPath)
    print('***'*30)
    print('load model config yaml:',configPath)
    print(model_yaml_config)
    print('***' * 30)

    # 保证可复现性
    reproducibility(model_yaml_config.seed)

    # 创建工作目录
    workdir = model_yaml_config.workdir
    if not os.path.exists(workdir): os.makedirs(workdir)

    training_log_path = os.path.join(workdir, '%s_training.log' % (current_time))
    logger = get_logger(training_log_path)
    logger.info("load config: %s" % (configPath))

    trainig_model_config_path = os.path.join(workdir,os.path.basename(default_model_config_path))
    shutil.copy(default_model_config_path,trainig_model_config_path)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 设置gpu
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    gpu_list = model_yaml_config.gpus
    gpu_list = [str(x) for x in gpu_list]
    gpu_list = ','.join(gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 加载模型结构
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    model_name = model_yaml_config.model_name
    model = create_model(model_config_args=model_yaml_config)

    ##

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 加载数据集
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_dataset = ISEG2017SampleDataset(
        csv_file_path=model_yaml_config['train_data_csv_path'],
        crop_size=model_yaml_config['crop_size'],
        mode='train',
        data_num=-1,
        use_aug=model_yaml_config['use_aug']
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=model_yaml_config.train_batch_size, num_workers=model_yaml_config.num_workers,pin_memory=True, shuffle=True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 选择损失函数
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    criterion = create_loss(model_config_args=model_yaml_config)


    # Define optimizers
    logger.info('opt: %s, lr_scheduler: %s' % (model_yaml_config.criterion_name,model_yaml_config.lr_scheduler))
    optimizer = None
    if model_yaml_config.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=model_yaml_config.opt_params['lr'],
                                     weight_decay=model_yaml_config.opt_params['weight_decay'])
    elif model_yaml_config.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                     lr=model_yaml_config.opt_params['lr'],
                                     weight_decay=model_yaml_config.opt_params['weight_decay'])

    elif model_yaml_config.opt == "adamw":
        print(f"weight decay argument will not be used. Default is 11e-2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_yaml_config.opt_params['lr'], weight_decay=model_yaml_config.opt_params['weight_decay'])

    elif model_yaml_config.opt == "ranger":
        from ranger import Ranger
        optimizer = Ranger(model.parameters(), lr=model_yaml_config.opt_params['lr'], weight_decay=model_yaml_config.opt_params['weight_decay'])

    else:
        logger.error(model_yaml_config.opt)
        assert False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # gpu parallel
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if len(gpu_list) > 0:
        model = model.cuda()
        criterion = criterion.cuda()
        model = torch.nn.DataParallel(model)  # multi-gpu training
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    logger.info('init model: %s' % (model_name))
    logger.info('using gpu: %s' % (str(gpu_list)))
    logger.info('select loss: %s' % (model_yaml_config.criterion_name))

    if model_yaml_config.lr_scheduler == 'StepLR':
        lr_scheduler = StepLR(optimizer, step_size=model_yaml_config.step_size, gamma=0.1)
    elif model_yaml_config.lr_scheduler == 'CosineAnnealingLR':
        MAX_STEP = int(1e10)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-5)
    elif model_yaml_config.lr_scheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)
    else:
        logger.error(model_yaml_config.lr_scheduler)
        assert False

    # 模型加载
    if model_yaml_config.resume:
        checkpoint = torch.load(model_yaml_config.resume_weight_path, map_location='cpu')
        saveEpoch = checkpoint['epoch']
        startEpoch = saveEpoch + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        logger.info("Resume Training! Save Epoch:%s, Now Start From %s"%(saveEpoch,startEpoch))
        logger.info("=> Loading checkpoint '{}'".format(model_yaml_config.resume_weight_path))
        logger.info("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        startEpoch = 0
        logger.info("New Training! Start From %s"%(startEpoch))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 训练开始
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    logger.info('start training!')
    # 早停
    not_improved_count = 0
    max_val_dice_score = 0
    #
    n_epoches = model_yaml_config.num_epochs
    n_iter_each_epoch = model_yaml_config['each_epoch_iter']
    all_train_iter= n_epoches * n_iter_each_epoch

    for epoch in range(startEpoch, n_epoches):
        logger.info('Epoch [%d/%d]' % (epoch, n_epoches))

        model.train()

        for iter_idx in tqdm(range(n_iter_each_epoch)):
            starttime = time.time()
            loss_seg_value1 = 0
            current_train_iter = epoch * n_iter_each_epoch + (iter_idx+1)

            feature_tensor, seg_gt_tensor = next(iter(train_loader))

            feature_inputs, seg_labels = feature_tensor.cuda(non_blocking=True), seg_gt_tensor.cuda(non_blocking=True)


            # ------------------------------------
            #  Train Generators (Segmentation)
            # ------------------------------------
            optimizer.zero_grad()

            # # # # # # # # # # # #
            # # train with source t1
            # # # # # # # # # # # #
            pred_outputs = model(feature_inputs)

            # # # # # # # # # # # #
            # # predict end
            # # # # # # # # # # # #

            loss_seg, per_ch_score = criterion(pred_outputs, seg_labels)

            # '[%s %.4f], ' % (self.label_names[i],channel_score[i])
            label_to_cate_dict = model_yaml_config.label_to_cate_dict
            dsc_score_result_log = ''
            for idx in range(len(per_ch_score)):
                dsc_score_result_log += '[%s: %.4f] ' % (label_to_cate_dict[str(idx)], per_ch_score[idx])

            loss_seg_all = loss_seg

            # backward
            loss_seg_all.backward()
            loss_seg_value1 += loss_seg_all.data.cpu().numpy()
            optimizer.step()

            # 释放显存
            del feature_inputs

            # multi gpu need gather vector
            lr = optimizer.param_groups[0]['lr']

            update_log = '[loss_seg1: %.3f], [lr: %.7f]' % (loss_seg_value1, lr)

            torch.cuda.empty_cache()
            newtime = time.time()
            if (iter_idx + 1) % (model_yaml_config.train_show_frep) == 0:
                logger.info('[epoch %d / %d], [iter_idx %d / %d], [train_iter %d / %d], %s, %s, [time %.3f]' %
                            (epoch,
                             n_epoches,
                             iter_idx + 1,
                             n_iter_each_epoch,
                             current_train_iter,
                             all_train_iter,
                             dsc_score_result_log,
                             update_log,
                             newtime - starttime))

        # 本次epoch中的评估情况
        logger.info('>>>>' * 30)
        logger.info('Evaluate Result:')
        # evaluate on validation set
        dice_csf, dice_gm, dice_wm, mean_dice = validate_by_patient(model)

        lr_scheduler.step(epoch=epoch)

        logger.info('[mode:val], [epoch %d], [dice_csf: %.4f], [dice_gm: %.4f], [dice_wm: %.4f]'%(epoch, dice_csf, dice_gm, dice_wm))
        logger.info('>>>>' * 30)


        if model_yaml_config.saveEpochs and (epoch + 1) % model_yaml_config.saveEpochs == 0:
            if mean_dice > max_val_dice_score:
                max_val_dice_score = mean_dice
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                filename = os.path.join(workdir, 'epoch_%s_val_dice_%s_csf_%s_gm_%s_wm_%s.pth.tar' % (epoch, mean_dice, dice_csf, dice_gm, dice_wm))
                logger.info('****' * 30)
                logger.info('Sava weight: %s'%(filename))
                logger.info('****' * 30)
                torch.save(state, filename)
                not_improved_count = 0
            else:
                not_improved_count += 1


        if not_improved_count > model_yaml_config.earlyStopEpoch:
            logger.info('>>>>' * 30)
            logger.info("Early Stop, Last Epoch:%s"%(epoch))
            logger.info('>>>>' * 30)
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
            filename = os.path.join(workdir, 'early_stop_%s_val_dsc_%s.pth.tar' % (epoch, mean_dice))
            torch.save(state, filename)
            break

        # torch.cuda.empty_cache()

    torch.cuda.empty_cache()
