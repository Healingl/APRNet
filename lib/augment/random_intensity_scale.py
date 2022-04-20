#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np


def random_intensity_scale(img_numpy,  scale_limits=(0.9, 1.1)):
    scale_range = scale_limits[1] - scale_limits[0]
    factor = scale_limits[0] + scale_range * np.random.random()
    img_numpy = img_numpy * factor

    return img_numpy


class RandomIntensityScale(object):
    def __init__(self, scale_limits=(0.9, 1.1)):
        self.scale_limits = scale_limits

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

        return random_intensity_scale(img_numpy, scale_limits=self.scale_limits), label
