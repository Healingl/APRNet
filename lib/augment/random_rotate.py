import numpy as np
import scipy.ndimage as ndimage


class RandomRotation(object):
    def __init__(self, angle_list=[90,180,270], axis=0):
        self.angles_idx_for_rotate = np.random.randint(0, len(angle_list))

        self.angle_list =angle_list
        self.axis = axis

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        rotate_angle = self.angle_list[self.angles_idx_for_rotate]
        # print('rotate_angle',rotate_angle)

        # 90度旋转
        assert rotate_angle % 90 == 0

        all_axes = [(1, 2), (0, 2), (0, 1)]
        rotate_axes = all_axes[self.axis]

        img_numpy = ndimage.rotate(img_numpy, rotate_angle, axes=rotate_axes)
        label = ndimage.rotate(label, rotate_angle, axes=rotate_axes)
        return img_numpy, label
