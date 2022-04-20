#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np


def random_intensity_shift(img_numpy, limit=0.1):
    shift_range = 2 * limit
    factor = -limit + shift_range * np.random.random()
    std = img_numpy.std()
    img_numpy= img_numpy + factor * std

    return img_numpy


class RandomIntensityShift(object):
    def __init__(self, limit=0.1):
        self.limit = limit

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be flipped. (155, 240, 240)
            label (numpy): Label segmentation map to be flipped (155, 240, 240)

        Returns:
            img_numpy (numpy):  flipped img.
            label (numpy): flipped Label segmentation.
        """
        assert len(img_numpy) > 0

        return random_intensity_shift(img_numpy, limit=self.limit), label
