import cv2
import random
import numpy as np
from ..core.composition import Compose
from ..core.transforms_interface import Transform, DualTransform
from . import functionals as F


class Float(DualTransform):
    def apply(self, img):
        return img.astype(np.float32)


class Contiguous(DualTransform):
    def apply(self, img):
        return np.ascontiguousarray(img)


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
    def __init__(self, var_limit=(0, 0.1), mean=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.var_limit = var_limit
        self.mean = mean

    def apply(self, img, var):
        return F.gaussian_noise(img, var=var, mean=self.mean)

    def get_params(self, **data):
        return {'var': random.uniform(*self.var_limit)}


class Resize(DualTransform):
    def __init__(self, shape, interpolation=3, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.shape = shape
        self.interpolation = interpolation

    def apply(self, img):
        return F.resize(img, new_shape=self.shape, interpolation=self.interpolation)
    
    def apply_to_mask(self, mask):
        return F.resize(mask, new_shape=self.shape, interpolation=0)


class RandomScale(DualTransform):
    def __init__(self, scale_limit=[0.9, 1.1], interpolation=3, always_apply=False, p=0.5):
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
    def __init__(self, scale_limit=[0.9, 1.1], interpolation=3, always_apply=False, p=0.5):
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
    def __init__(self, axes=(0,1), limit=(-90, 90), interpolation=3, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
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
    def __init__(self, axes=(0,1), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes

    def apply(self, img, factor):
        return np.rot90(img, factor, axes=self.axes)

    def get_params(self, **data):
        return {"factor": random.randint(0, 3)}


class Flip(DualTransform):
    def __init__(self, axis=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axis = axis

    def apply(self, img):
        return np.flip(img, self.axis)

    
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
    def __init__(self, shape, scale_limit=(0.8, 1.2), interpolation=3, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.shape = shape
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def apply(self, img, scale=1, scaled_shape=None, h_start=0, w_start=0, d_start=0):
        if scaled_shape is None:
            scaled_shape = self.shape
        img = F.random_crop(img, scaled_shape[0], scaled_shape[1], scaled_shape[2], h_start, w_start, d_start)
        return F.resize(img, new_shape=self.shape, interpolation=self.interpolation)

    def apply_to_mask(self, img, scale=1, scaled_shape=None, h_start=0, w_start=0, d_start=0):
        if scaled_shape is None:
            scaled_shape = self.shape
        img = F.random_crop(img, scaled_shape[0], scaled_shape[1], scaled_shape[2], h_start, w_start, d_start)
        return F.resize(img, new_shape=self.shape, interpolation=0)

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
    def __init__(self, shape, scale_limit=(0.8, 1.2), interpolation=3, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.shape = shape
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def apply(self, img, x_min=0, y_min=0, z_min=0, x_max=0, y_max=0, z_max=0):
        img = F.crop(img, x_min, y_min, z_min, x_max, y_max, z_max)
        return F.resize(img, self.shape, interpolation=self.interpolation)

    def apply_to_mask(self, img, x_min=0, y_min=0, z_min=0, x_max=0, y_max=0, z_max=0):
        img = F.crop(img, x_min, y_min, z_min, x_max, y_max, z_max)
        return F.resize(img, self.shape, interpolation=0)

    def get_params(self, **data):
        mask = data["mask"] # [H, W, D]
        mask_height, mask_width, mask_depth = mask.shape
        
        scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
        height, width, depth = [int(scale * i) for i in self.shape]

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


class RandomGamma(Transform):
    def __init__(self, gamma_limit=(0.7, 1.5), eps=1e-7, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.gamma_limit = gamma_limit
        self.eps = eps

    def apply(self, img, gamma=1):
        return F.gamma_transform(img, gamma=gamma, eps=self.eps)

    def get_params(self, **data):
        return {"gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1])}


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
    def __init__(self, deformation_limits=(0, 0.25), interpolation=3, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
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
    def __init__(self, x_limit=(-15,15), y_limit=(-15,15), z_limit=(-15,15), interpolation=3, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
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
