#=================================================================================#
#  Author:       Pavel Iakubovskii, ZFTurbo, ashawkey, Dominik Müller             #
#  Copyright:    albumentations:    : https://github.com/albumentations-team      #
#                Pavel Iakubovskii  : https://github.com/qubvel                   #
#                ZFTurbo            : https://github.com/ZFTurbo                  #
#                ashawkey           : https://github.com/ashawkey                 #
#                Dominik Müller     : https://github.com/muellerdo                #
#                                                                                 #
#  Volumentations History:                                                        #
#       - Original:                 https://github.com/albumentations-team/album  #
#                                   entations                                     #
#       - 3D Conversion:            https://github.com/ashawkey/volumentations    #
#       - Continued Development:    https://github.com/ZFTurbo/volumentations     #
#       - Enhancements:             https://github.com/qubvel/volumentations      #
#       - Further Enhancements:     https://github.com/muellerdo/volumentations   #
#                                                                                 #
#  MIT License.                                                                   #
#                                                                                 #
#  Permission is hereby granted, free of charge, to any person obtaining a copy   #
#  of this software and associated documentation files (the "Software"), to deal  #
#  in the Software without restriction, including without limitation the rights   #
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
#  copies of the Software, and to permit persons to whom the Software is          #
#  furnished to do so, subject to the following conditions:                       #
#                                                                                 #
#  The above copyright notice and this permission notice shall be included in all #
#  copies or substantial portions of the Software.                                #
#                                                                                 #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
#  SOFTWARE.                                                                      #
#=================================================================================#
import cv2
import random
import numpy as np
import numbers
from enum import Enum, IntEnum
from ..core.transforms_interface import *
from ..augmentations import functional as F
from ..random_utils import *

class Float(DualTransform):
    def apply(self, image):
        return image.astype(np.float32)

class Contiguous(DualTransform):
    def apply(self, image):
        return np.ascontiguousarray(image)


class PadIfNeeded(DualTransform):
    def __init__(self, shape, border_mode='constant', value=0, mask_value=0, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.shape = shape
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img):
        return F.pad(img, self.shape, self.border_mode, self.value)

    def apply_to_mask(self, mask):
        return F.pad(mask, self.shape, self.border_mode, self.mask_value)


class GaussianNoise(Transform):
    def __init__(self, var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.var_limit = var_limit
        self.mean = mean

    def apply(self, img, gauss=None):
        return F.gaussian_noise(img, gauss=gauss)

    def get_params(self, **data):
        image = data["image"]
        var = uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5

        gauss = normal(self.mean, sigma, image.shape)
        return {"gauss": gauss}


class Resize(DualTransform):
    def __init__(self, shape, interpolation=1, resize_type=1, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.shape = shape
        self.interpolation = interpolation
        self.resize_type = resize_type

    def apply(self, img):
        return F.resize(img, new_shape=self.shape, interpolation=self.interpolation, resize_type=self.resize_type)

    def apply_to_mask(self, mask):
        return F.resize(mask, new_shape=self.shape, interpolation=0, resize_type=self.resize_type)


class RandomScale(DualTransform):
    def __init__(self, scale_limit=[0.9, 1.1], interpolation=1, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def get_params(self, **data):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale):
        return F.rescale(img, scale, interpolation=self.interpolation)

    def apply_to_mask(self, mask, scale):
        return F.rescale(mask, scale, interpolation=0)


class RandomScale2(DualTransform):
    """
    TODO: compare speeds with version 1.
    """
    def __init__(self, scale_limit=[0.9, 1.1], interpolation=1, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def get_params(self, **data):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale):
        return F.rescale_warp(img, scale, interpolation=self.interpolation)

    def apply_to_mask(self, mask, scale):
        return F.rescale_warp(mask, scale, interpolation=0)


class RotatePseudo2D(DualTransform):
    def __init__(self, axes=(0,1), limit=(-90, 90), interpolation=1, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle):
        return F.rotate2d(img, angle, axes=self.axes, reshape=False, interpolation=self.interpolation, border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, mask, angle):
        return F.rotate2d(mask, angle, axes=self.axes, reshape=False, interpolation=0, border_mode=self.border_mode, value=self.mask_value)

    def get_params(self, **data):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}


class RandomRotate90(DualTransform):
    def __init__(self, axes=None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes

    def apply(self, img, axes, factor):
        return np.rot90(img, factor, axes=axes)

    def get_params(self, **data):
        # Pick predefined axis to flip
        if self.axes is not None : axes = self.axes
        # Pick random combination of axes to flip
        else:
            combinations = [(0,1), (1,0), (0,2), (2,0), (1,2), (2,1)]
            axes = random.choice(combinations)
        # Define params
        return {"factor": random.randint(0, 3),
                "axes": axes}


class Flip(DualTransform):
    def __init__(self, axis=None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axis = axis

    def apply(self, img):
        # Pick predefined axis to flip
        if self.axis is not None : axis = self.axis
        # Pick random combination of axes to flip
        else:
            combinations = [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
            axis = random.choice(combinations)
        # Apply flipping
        return np.flip(img, axis)


class Normalize(Transform):
    def __init__(self, range_norm=False, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.range_norm = range_norm

    def apply(self, img):
        return F.normalize(img, range_norm=self.range_norm)


class Transpose(DualTransform):
    def __init__(self, axes=(1,0,2), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes

    def apply(self, img):
        return np.transpose(img, self.axes)


class CenterCrop(DualTransform):
    def __init__(self, shape, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.shape = shape

    def apply(self, img):
        return F.center_crop(img, self.shape[0], self.shape[1], self.shape[2])


class RandomResizedCrop(DualTransform):
    def __init__(self, shape, scale_limit=(0.8, 1.2), interpolation=1, resize_type=1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.shape = shape
        self.scale_limit = scale_limit
        self.interpolation = interpolation
        self.resize_type = resize_type

    def apply(self, img, scale=1, scaled_shape=None, h_start=0, w_start=0, d_start=0):
        if scaled_shape is None:
            scaled_shape = self.shape
        img = F.random_crop(img, scaled_shape[0], scaled_shape[1], scaled_shape[2], h_start, w_start, d_start)
        return F.resize(img, new_shape=self.shape, interpolation=self.interpolation, resize_type=self.resize_type)

    def apply_to_mask(self, img, scale=1, scaled_shape=None, h_start=0, w_start=0, d_start=0):
        if scaled_shape is None:
            scaled_shape = self.shape
        img = F.random_crop(img, scaled_shape[0], scaled_shape[1], scaled_shape[2], h_start, w_start, d_start)
        return F.resize(img, new_shape=self.shape, interpolation=0, resize_type=self.resize_type)

    def get_params(self, **data):
        scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
        scaled_shape = [int(scale * i) for i in self.shape]
        return {
            "scale": scale,
            "scaled_shape": scaled_shape,
            "h_start": random.random(),
            "w_start": random.random(),
            "d_start": random.random(),
        }


class RandomCrop(DualTransform):
    def __init__(self, shape, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.shape = shape

    def apply(self, img, h_start=0, w_start=0, d_start=0):
        return F.random_crop(img, self.shape[0], self.shape[1], self.shape[2], h_start, w_start, d_start)

    def get_params(self, **data):
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "d_start": random.random(),
        }


class CropNonEmptyMaskIfExists(DualTransform):
    def __init__(self, shape, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = shape[0]
        self.width = shape[1]
        self.depth = shape[2]

    def apply(self, img, x_min=0, y_min=0, z_min=0, x_max=0, y_max=0, z_max=0):
        return F.crop(img, x_min, y_min, z_min, x_max, y_max, z_max)

    def get_params(self, **data):
        mask = data["mask"] # [H, W, D]
        mask_height, mask_width, mask_depth = mask.shape

        if mask.sum() == 0:
            x_min = random.randint(0, mask_height - self.height)
            y_min = random.randint(0, mask_width - self.width)
            z_min = random.randint(0, mask_depth - self.depth)
        else:
            non_zero = np.argwhere(mask)
            x, y, z = random.choice(non_zero)
            x_min = x - random.randint(0, self.height - 1)
            y_min = y - random.randint(0, self.width - 1)
            z_min = z - random.randint(0, self.depth - 1)
            x_min = np.clip(x_min, 0, mask_height - self.height)
            y_min = np.clip(y_min, 0, mask_width - self.width)
            z_min = np.clip(z_min, 0, mask_depth - self.depth)

        x_max = x_min + self.height
        y_max = y_min + self.width
        z_max = z_min + self.depth

        return {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max,
        }


class ResizedCropNonEmptyMaskIfExists(DualTransform):
    def __init__(self, shape, scale_limit=(0.8, 1.2), interpolation=1, resize_type=1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.shape = shape
        self.scale_limit = scale_limit
        self.interpolation = interpolation
        self.resize_type = resize_type

    def apply(self, img, x_min=0, y_min=0, z_min=0, x_max=0, y_max=0, z_max=0):
        img = F.crop(img, x_min, y_min, z_min, x_max, y_max, z_max)
        return F.resize(img, self.shape, interpolation=self.interpolation, resize_type=self.resize_type)

    def apply_to_mask(self, img, x_min=0, y_min=0, z_min=0, x_max=0, y_max=0, z_max=0):
        img = F.crop(img, x_min, y_min, z_min, x_max, y_max, z_max)
        return F.resize(img, self.shape, interpolation=0, resize_type=self.resize_type)

    def get_params(self, **data):
        mask = data["mask"] # [H, W, D]
        mask_height, mask_width, mask_depth = mask.shape

        scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
        height, width, depth = [int(scale * i) for i in self.shape]

        if mask.sum() == 0:
            x_min = random.randint(0, mask_height - height)
            y_min = random.randint(0, mask_width - width)
            z_min = random.randint(0, mask_depth - depth)
        else:
            non_zero = np.argwhere(mask)
            x, y, z = random.choice(non_zero)
            x_min = x - random.randint(0, height - 1)
            y_min = y - random.randint(0, width - 1)
            z_min = z - random.randint(0, depth - 1)
            x_min = np.clip(x_min, 0, mask_height - height)
            y_min = np.clip(y_min, 0, mask_width - width)
            z_min = np.clip(z_min, 0, mask_depth - depth)

        x_max = x_min + height
        y_max = y_min + width
        z_max = z_min + depth

        return {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max,
        }


class RandomGamma(ImageOnlyTransform):
    """
    Args:
        gamma_limit (float or (float, float)): If gamma_limit is a single float value,
            the range will be (-gamma_limit, gamma_limit). Default: (80, 120).
        eps: Deprecated.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = to_tuple(gamma_limit)
        self.eps = eps

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self, **data):
        return {"gamma": random.randint(self.gamma_limit[0], self.gamma_limit[1]) / 100.0}

    def get_transform_init_args_names(self):
        return ("gamma_limit", "eps")


class ElasticTransformPseudo2D(DualTransform):
    def __init__(self, alpha=1000, sigma=50, alpha_affine=1, approximate=False, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.approximate = approximate

    def apply(self, img, random_state=None):
        return F.elastic_transform_pseudo2D(img, self.alpha, self.sigma, self.alpha_affine, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None, random_state=random_state, approximate=False)

    def apply_to_mask(self, img, random_state=None):
        return F.elastic_transform_pseudo2D(img, self.alpha, self.sigma, self.alpha_affine, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101, value=None, random_state=random_state, approximate=False)

    def get_params(self, **data):
        return {"random_state": random.randint(0, 10000)}


class ElasticTransform(DualTransform):
    def __init__(self, deformation_limits=(0, 0.25), interpolation=1, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.deformation_limits = deformation_limits
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, sigmas, alphas, random_state=None):
        return F.elastic_transform(img, sigmas, alphas, interpolation=self.interpolation, random_state=random_state, border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, img, sigmas, alphas, random_state=None):
        return F.elastic_transform(img, sigmas, alphas, interpolation=0, random_state=random_state, border_mode=self.border_mode, value=self.mask_value)

    def get_params(self, **data):
        image = data["image"] # [H, W, D]
        random_state = random.randint(0, 10000)
        deformation = random.uniform(*self.deformation_limits)
        sigmas = [deformation * x for x in image.shape[:3]]
        alphas = [random.uniform(x/8, x/2) for x in sigmas]
        return {
            "random_state": random_state,
            "sigmas": sigmas,
            "alphas": alphas,
        }


class Rotate(DualTransform):
    def __init__(self, x_limit=(-15,15), y_limit=(-15,15), z_limit=(-15,15), interpolation=1, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.z_limit = z_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, x, y, z):
        return F.rotate3d(img, x, y, z, interpolation=self.interpolation, border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, mask, x, y, z):
        return F.rotate3d(mask, x, y, z, interpolation=0, border_mode=self.border_mode, value=self.mask_value)

    def get_params(self, **data):
        return {
            "x": random.uniform(self.x_limit[0], self.x_limit[1]),
            "y": random.uniform(self.y_limit[0], self.y_limit[1]),
            "z": random.uniform(self.z_limit[0], self.z_limit[1]),
        }


class RemoveEmptyBorder(DualTransform):
    def __init__(self, border_value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.border_value = border_value


    def apply(self, img, x_min=0, y_min=0, z_min=0, x_max=0, y_max=0, z_max=0):
        return F.crop(img, x_min, y_min, z_min, x_max, y_max, z_max)

    def get_params(self, **data):
        image = data["image"] # [H, W, D, C]

        borders = np.where(image != self.border_value)

        return {
            "x_min": np.min(borders[0]), "x_max": np.max(borders[0])+1,
            "y_min": np.min(borders[1]), "y_max": np.max(borders[1])+1,
            "z_min": np.min(borders[2]), "z_max": np.max(borders[2])+1,
        }


class RandomCropFromBorders(DualTransform):
    """Crop bbox from image randomly cut parts from borders without resize at the end

    Args:
        crop_value (float): float value in (0.0, 0.5) range. Default 0.1
        crop_0_min (float): float value in (0.0, 1.0) range. Default 0.1
        crop_0_max (float): float value in (0.0, 1.0) range. Default 0.1
        crop_1_min (float): float value in (0.0, 1.0) range. Default 0.1
        crop_1_max (float): float value in (0.0, 1.0) range. Default 0.1
        crop_2_min (float): float value in (0.0, 1.0) range. Default 0.1
        crop_2_max (float): float value in (0.0, 1.0) range. Default 0.1
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            crop_value=None,
            crop_0_min=None,
            crop_0_max=None,
            crop_1_min=None,
            crop_1_max=None,
            crop_2_min=None,
            crop_2_max=None,
            always_apply=False,
            p=1.0
    ):
        super(RandomCropFromBorders, self).__init__(always_apply, p)
        self.crop_0_min = 0.1
        self.crop_0_max = 0.1
        self.crop_1_min = 0.1
        self.crop_1_max = 0.1
        self.crop_2_min = 0.1
        self.crop_2_max = 0.1
        if crop_value is not None:
            self.crop_0_min = crop_value
            self.crop_0_max = crop_value
            self.crop_1_min = crop_value
            self.crop_1_max = crop_value
            self.crop_2_min = crop_value
            self.crop_2_max = crop_value
        if crop_0_min is not None:
            self.crop_0_min = crop_0_min
        if crop_0_max is not None:
            self.crop_0_max = crop_0_max
        if crop_1_min is not None:
            self.crop_1_min = crop_1_min
        if crop_1_max is not None:
            self.crop_1_max = crop_1_max
        if crop_2_min is not None:
            self.crop_2_min = crop_2_min
        if crop_2_max is not None:
            self.crop_2_max = crop_2_max

    def get_params(self, **data):
        img = data["image"]
        sh0_min = random.randint(0, int(self.crop_0_min * img.shape[0]))
        sh0_max = random.randint(max(sh0_min + 1, int((1 - self.crop_0_max) * img.shape[0])), img.shape[0])

        sh1_min = random.randint(0, int(self.crop_1_min * img.shape[1]))
        sh1_max = random.randint(max(sh1_min + 1, int((1 - self.crop_1_max) * img.shape[1])), img.shape[1])

        sh2_min = random.randint(0, int(self.crop_2_min * img.shape[2]))
        sh2_max = random.randint(max(sh2_min + 1, int((1 - self.crop_2_max) * img.shape[2])), img.shape[2])

        return {
            "sh0_min": sh0_min, "sh0_max": sh0_max,
            "sh1_min": sh1_min, "sh1_max": sh1_max,
            "sh2_min": sh2_min, "sh2_max": sh2_max
        }

    def apply(self, img, sh0_min=0, sh0_max=0, sh1_min=0, sh1_max=0, sh2_min=0, sh2_max=0, **params):
        return F.clamping_crop(img, sh0_min, sh1_min, sh2_min, sh0_max, sh1_max, sh2_max)

    def apply_to_mask(self, mask, sh0_min=0, sh0_max=0, sh1_min=0, sh1_max=0, sh2_min=0, sh2_max=0, **params):
        return F.clamping_crop(mask, sh0_min, sh1_min, sh2_min, sh0_max, sh1_max, sh2_max)


class GridDropout(DualTransform):
    """GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
    Args:
        ratio (float): the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_min (int): minimum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        unit_size_max (int): maximum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        holes_number_x (int): the number of grid units in x direction. Must be between 1 and image width//2.
            If 'None', grid unit width is set as image_width//10. Default: `None`.
        holes_number_y (int): the number of grid units in y direction. Must be between 1 and image height//2.
            If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
        holes_number_z (int): the number of grid units in z direction. Must be between 1 and image depth//2.
            If `None`, grid unit depth is set equal to the grid unit width or image height, whatever is smaller.
        shift_x (int): offsets of the grid start in x direction from (0,0) coordinate.
            Clipped between 0 and grid unit_width - hole_width. Default: 0.
        shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
            Clipped between 0 and grid unit height - hole_height. Default: 0.
        shift_z (int): offsets of the grid start in z direction from (0,0) coordinate.
            Clipped between 0 and grid unit depth - hole_depth. Default: 0.
        random_offset (boolean): weather to offset the grid randomly between 0 and grid unit size - hole size
            If 'True', entered shift_x, shift_y, shift_z are ignored and set randomly. Default: `False`.
        fill_value (int): value for the dropped pixels. Default = 0
        mask_fill_value (int): value for the dropped pixels in mask.
            If `None`, tranformation is not applied to the mask. Default: `None`.
    Targets:
        image, mask
    Image types:
        uint8, float32
    References:
        https://arxiv.org/abs/2001.04086
    """

    def __init__(
        self,
        ratio: float = 0.5,
        unit_size_min: int = None,
        unit_size_max: int = None,
        holes_number_x: int = None,
        holes_number_y: int = None,
        holes_number_z: int = None,
        shift_x: int = 0,
        shift_y: int = 0,
        shift_z: int = 0,
        random_offset: bool = False,
        fill_value: int = 0,
        mask_fill_value: int = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(GridDropout, self).__init__(always_apply, p)
        self.ratio = ratio
        self.unit_size_min = unit_size_min
        self.unit_size_max = unit_size_max
        self.holes_number_x = holes_number_x
        self.holes_number_y = holes_number_y
        self.holes_number_z = holes_number_z
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be between 0 and 1.")

    def apply(self, image, holes=(), **params):
        return F.cutout(image, holes, self.fill_value)

    def apply_to_mask(self, image, holes=(), **params):
        if self.mask_fill_value is None:
            return image

        return F.cutout(image, holes, self.mask_fill_value)

    def get_params(self, **data):
        img = data["image"]
        height, width, depth = img.shape[:3]
        # set grid using unit size limits
        if self.unit_size_min and self.unit_size_max:
            if not 2 <= self.unit_size_min <= self.unit_size_max:
                raise ValueError("Max unit size should be >= min size, both at least 2 pixels.")
            if self.unit_size_max > min(height, width):
                raise ValueError("Grid size limits must be within the shortest image edge.")
            unit_width = random.randint(self.unit_size_min, self.unit_size_max + 1)
            unit_height = unit_width
            unit_depth = unit_width
        else:
            # set grid using holes numbers
            if self.holes_number_x is None:
                unit_width = max(2, width // 10)
            else:
                if not 1 <= self.holes_number_x <= width // 2:
                    raise ValueError("The hole_number_x must be between 1 and image width//2.")
                unit_width = width // self.holes_number_x
            if self.holes_number_y is None:
                unit_height = max(min(unit_width, height), 2)
            else:
                if not 1 <= self.holes_number_y <= height // 2:
                    raise ValueError("The hole_number_y must be between 1 and image height//2.")
                unit_height = height // self.holes_number_y
            if self.holes_number_z is None:
                unit_depth = max(min(unit_height, depth), 2)
            else:
                if not 1 <= self.holes_number_z <= depth // 2:
                    raise ValueError("The hole_number_z must be between 1 and image depth//2.")
                unit_depth = depth // self.holes_number_z

        hole_width = int(unit_width * self.ratio)
        hole_height = int(unit_height * self.ratio)
        hole_depth = int(unit_depth * self.ratio)
        # min 1 pixel and max unit length - 1
        hole_width = min(max(hole_width, 1), unit_width - 1)
        hole_height = min(max(hole_height, 1), unit_height - 1)
        hole_depth = min(max(hole_depth, 1), unit_depth - 1)
        # set offset of the grid
        if self.shift_x is None:
            shift_x = 0
        else:
            shift_x = min(max(0, self.shift_x), unit_width - hole_width)
        if self.shift_y is None:
            shift_y = 0
        else:
            shift_y = min(max(0, self.shift_y), unit_height - hole_height)
        if self.shift_z is None:
            shift_z = 0
        else:
            shift_z = min(max(0, self.shift_z), unit_depth - hole_depth)
        if self.random_offset:
            shift_x = random.randint(0, unit_width - hole_width)
            shift_y = random.randint(0, unit_height - hole_height)
            shift_z = random.randint(0, unit_depth - hole_depth)
        holes = []
        for i in range(width // unit_width + 1):
            for j in range(height // unit_height + 1):
                for k in range(depth // unit_depth + 1):
                    x1 = min(shift_x + unit_width * i, width)
                    y1 = min(shift_y + unit_height * j, height)
                    z1 = min(shift_z + unit_depth * j, depth)
                    x2 = min(x1 + hole_width, width)
                    y2 = min(y1 + hole_height, height)
                    z2 = min(z1 + hole_depth, depth)
                    holes.append((x1, y1, z1, x2, y2, z2))

        return {"holes": holes}

    def get_transform_init_args_names(self):
        return (
            "ratio",
            "unit_size_min",
            "unit_size_max",
            "holes_number_x",
            "holes_number_y",
            "shift_x",
            "shift_y",
            "mask_fill_value",
            "random_offset",
        )


class RandomDropPlane(DualTransform):
    """Randomly drop some planes in random axis

    Args:
        plane_drop_prob (float): float value in (0.0, 1.0) range. Default: 0.1
        axes (tuple). Default: 0
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            plane_drop_prob=0.1,
            axes=(0,),
            always_apply=False,
            p=1.0
    ):
        super(RandomDropPlane, self).__init__(always_apply, p)
        self.plane_drop_prob = plane_drop_prob
        self.axes = axes

    def get_params(self, **data):
        img = data["image"]
        axis = random.choice(self.axes)
        r = img.shape[axis]
        indexes = []
        for i in range(r):
            if random.uniform(0, 1) > self.plane_drop_prob:
                indexes.append(i)
        if len(indexes) == 0:
            indexes.append(0)

        return {
            "indexes": indexes, "axis": axis,
        }

    def apply(self, img, indexes=(), axis=0, **params):
        return np.take(img, indexes, axis=axis)

    def apply_to_mask(self, mask, indexes=(), axis=0, **params):
        return np.take(mask, indexes, axis=axis)

class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.
    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
    ):
        super(RandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self, **data):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit", "brightness_by_max")


class ColorJitter(ImageOnlyTransform):
    """Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
    this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
    Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
    overflow, but we use value saturation.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
        always_apply=False,
        p=0.5,
    ):
        super(ColorJitter, self).__init__(always_apply=always_apply, p=p)

        self.brightness = self.__check_values(brightness, "brightness")
        self.contrast = self.__check_values(contrast, "contrast")
        self.saturation = self.__check_values(saturation, "saturation")
        self.hue = self.__check_values(hue, "hue", offset=0, bounds=[-0.5, 0.5], clip=False)

    @staticmethod
    def __check_values(value, name, offset=1, bounds=(0, float("inf")), clip=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [offset - value, offset + value]
            if clip:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError("{} values should be between {}".format(name, bounds))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        return value

    def get_params(self, **data):
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])
        hue = random.uniform(self.hue[0], self.hue[1])

        transforms = [
            lambda x: F.adjust_brightness_torchvision(x, brightness),
            lambda x: F.adjust_contrast_torchvision(x, contrast),
            lambda x: F.adjust_saturation_torchvision(x, saturation),
            lambda x: F.adjust_hue_torchvision(x, hue),
        ]
        random.shuffle(transforms)

        return {"transforms": transforms}

    def apply(self, img, transforms=(), **params):
        if not F.is_3Drgb_image(img) and not F.is_3Dgrayscale_image(img):
            raise TypeError("ColorJitter transformation expects 1-channel or 3-channel images.")

        for transform in transforms:
            img_transformed = np.zeros(img.shape, dtype=np.float32)
            for slice in range(img.shape[0]):
                img_transformed[slice,:,:] = transform(img[slice,:,:].astype(np.float32))
        return img_transformed

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "saturation", "hue")


class GridDistortion(DualTransform):
    """
    Args:
        num_steps (int): count of grid cells on each side.
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        num_steps=5,
        distort_limit=0.3,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(GridDistortion, self).__init__(always_apply, p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        img_transformed = np.zeros(img.shape, dtype=img.dtype)
        for slice in range(img.shape[0]):
            img_transformed[slice,:,:] = F.grid_distortion(img[slice,:,:],
                                                           self.num_steps,
                                                           stepsx,
                                                           stepsy,
                                                           interpolation,
                                                           self.border_mode,
                                                           self.value)
        return img_transformed

    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        img_transformed = np.zeros(img.shape, dtype=img.dtype)
        for slice in range(img.shape[0]):
            img_transformed[slice,:,:] = F.grid_distortion(img[slice,:,:],
                                                           self.num_steps,
                                                           stepsx,
                                                           stepsy,
                                                           cv2.INTER_NEAREST,
                                                           self.border_mode,
                                                           self.mask_value)
        return img_transformed

    def get_params(self, **data):
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        return {"stepsx": stepsx, "stepsy": stepsy}

    def get_transform_init_args_names(self):
        return (
            "num_steps",
            "distort_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )

class Downscale(ImageOnlyTransform):
    """Decreases image quality by downscaling and upscaling back.
    Args:
        scale_min (float): lower bound on the image scale. Should be < 1.
        scale_max (float):  lower bound on the image scale. Should be .
        interpolation: cv2 interpolation method. cv2.INTER_NEAREST by default
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale_min=0.25,
        scale_max=0.25,
        interpolation=cv2.INTER_NEAREST,
        always_apply=False,
        p=0.5,
    ):
        super(Downscale, self).__init__(always_apply, p)
        if scale_min > scale_max:
            raise ValueError("Expected scale_min be less or equal scale_max, got {} {}".format(scale_min, scale_max))
        if scale_max >= 1:
            raise ValueError("Expected scale_max to be less than 1, got {}".format(scale_max))
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interpolation = interpolation

    def apply(self, image, scale, interpolation, **params):
        return F.downscale(image, scale=scale, interpolation=interpolation)

    def get_params(self, **data):
        return {
            "scale": random.uniform(self.scale_min, self.scale_max),
            "interpolation": self.interpolation,
        }

    def get_transform_init_args_names(self):
        return "scale_min", "scale_max", "interpolation"

class GlassBlur(Blur):
    """Apply glass noise to the input image.
    Args:
        sigma (float): standard deviation for Gaussian kernel.
        max_delta (int): max distance between pixels which are swapped.
        iterations (int): number of repeats.
            Should be in range [1, inf). Default: (2).
        mode (str): mode of computation: fast or exact. Default: "fast".
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    Reference:
    |  https://arxiv.org/abs/1903.12261
    |  https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    """

    def __init__(
        self,
        sigma=0.7,
        max_delta=4,
        iterations=2,
        always_apply=False,
        mode="fast",
        p=0.5,
    ):
        super(GlassBlur, self).__init__(always_apply=always_apply, p=p)
        if iterations < 1:
            raise ValueError("Iterations should be more or equal to 1, but we got {}".format(iterations))

        if mode not in ["fast", "exact"]:
            raise ValueError("Mode should be 'fast' or 'exact', but we got {}".format(iterations))

        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations
        self.mode = mode

    def apply(self, img, dxy, **data):
        img_blurred = np.zeros(img.shape, dtype=img.dtype)
        for slice in range(img.shape[0]):
            img_processed = F.glass_blur(img[slice,:,:],
                                         self.sigma,
                                         self.max_delta,
                                         self.iterations,
                                         dxy,
                                         self.mode)
            if len(img.shape) == 4 and img.shape[-1] == 1:
                img_blurred[slice,:,:] = np.reshape(img_processed,
                                                        img_processed.shape+(1,))
            else : img_blurred[slice,:,:] = img_processed
        return img_blurred

    def get_params(self, **data):
        img = data["image"]

        # generate array containing all necessary values for transformations
        width_pixels = img.shape[1] - self.max_delta * 2
        height_pixels = img.shape[2] - self.max_delta * 2
        total_pixels = width_pixels * height_pixels
        dxy = randint(-self.max_delta, self.max_delta, size=(total_pixels, self.iterations, 2))

        return {"dxy": dxy}

    def get_transform_init_args_names(self):
        return ("sigma", "max_delta", "iterations")

    @property
    def targets_as_params(self):
        return ["image"]

class ImageCompression(ImageOnlyTransform):
    """Decrease Jpeg, WebP compression of an image.
    Args:
        quality_lower (float): lower bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper (float): upper bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG
    Targets:
        image
    Image types:
        uint8, float32
    """

    class ImageCompressionType(IntEnum):
        JPEG = 0
        WEBP = 1

    def __init__(
        self,
        quality_lower=99,
        quality_upper=100,
        compression_type=ImageCompressionType.JPEG,
        always_apply=False,
        p=0.5,
    ):
        super(ImageCompression, self).__init__(always_apply, p)

        self.compression_type = ImageCompression.ImageCompressionType(compression_type)
        low_thresh_quality_assert = 0

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            low_thresh_quality_assert = 1

        if not low_thresh_quality_assert <= quality_lower <= 100:
            raise ValueError("Invalid quality_lower. Got: {}".format(quality_lower))
        if not low_thresh_quality_assert <= quality_upper <= 100:
            raise ValueError("Invalid quality_upper. Got: {}".format(quality_upper))

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, img, quality=100, image_type=".jpg", **params):
        if not F.is_3Drgb_image(img) and not F.is_3Dgrayscale_image(img):
            raise TypeError("ImageCompression transformation expects 1, 3 or 4 channel images.")


        img_transformed = np.zeros(img.shape, dtype=img.dtype)
        for slice in range(img.shape[0]):
            img_transformed[slice,:,:] = F.image_compression(img[slice,:,:],
                                                             quality,
                                                             image_type)
        return img_transformed

    def get_params(self, **data):
        image_type = ".jpg"

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            image_type = ".webp"

        return {
            "quality": random.randint(self.quality_lower, self.quality_upper),
            "image_type": image_type,
        }

    def get_transform_init_args(self):
        return {
            "quality_lower": self.quality_lower,
            "quality_upper": self.quality_upper,
            "compression_type": self.compression_type.value,
        }
