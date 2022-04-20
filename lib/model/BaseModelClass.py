"""
Implementation of BaseModel taken and modified from here
https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/basemodel/basemodel.py
"""

import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    r"""
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def test(self):
        """
        To be implemented by the subclass so that
        models can perform a forward propagation
        :return:
        """
        pass

    def count_params(self,cal_as_m=True):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        if cal_as_m:
            num_total_params = num_total_params / 1e6
            num_trainable_params = num_trainable_params / 1e6
            print('total parameters: %.2fM, total trainable params: %.2fM'%(num_total_params,num_trainable_params))
        return num_total_params, num_trainable_params