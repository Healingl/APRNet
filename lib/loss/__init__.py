import torch
import torch.nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

from .BCEDice import BCEDiceLoss
from torch.nn import CrossEntropyLoss

def create_loss(model_config_args):
    support_loss_list = ['BCEDiceLoss',
                        ]

    loss_name = model_config_args.criterion_name

    if loss_name not in support_loss_list:
        print(support_loss_list)
        assert False

    print('--' * 50)
    print('Choose Loss Name: %s' % (loss_name))

    if loss_name == 'BCEDiceLoss':
        criterion = BCEDiceLoss()
    else:
        assert False

    # assert (loss.item() > 0) and (loss.item() < 3)
    print("%s is ok!"%(loss_name))
    print('--' * 50)

    return criterion

