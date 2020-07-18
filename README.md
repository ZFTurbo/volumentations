# Volumentations

Fork of [volumentations](https://github.com/ashawkey/volumentations) 3D Volume data augmentation package by [@ashawkey](https://github.com/ashawkey/).
Initially inspired by [albumentations](https://github.com/albumentations-team/albumentations) library for augmentation of 2D images.

### Simple Example

```python
from volumentations import *

def get_augmentation(patch_size):
    return Compose([
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        RandomCropFromBorders(crop_value=0.1, p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.2),
        RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)

aug = get_augmentation((64, 128, 128))

data = {'image': img, 'mask': lbl}
aug_data = aug(**data)
img, lbl = aug_data['image'], aug_data['mask']
```

### Implemented 3D augmentations

```python
class PadIfNeeded(DualTransform):
class GaussianNoise(Transform):
class Resize(DualTransform):
class RandomScale(DualTransform):
class RotatePseudo2D(DualTransform):
class RandomRotate90(DualTransform):
class Flip(DualTransform):
class Normalize(Transform):
class Float(DualTransform):
class Contiguous(DualTransform):
class Transpose(DualTransform):
class CenterCrop(DualTransform):
class RandomResizedCrop(DualTransform):
class RandomCrop(DualTransform):
class CropNonEmptyMaskIfExists(DualTransform):
class ResizedCropNonEmptyMaskIfExists(DualTransform):
class RandomGamma(Transform):
class ElasticTransformPseudo2D(DualTransform):
class ElasticTransform(DualTransform):
class Rotate(DualTransform):
class RandomCropFromBorders(DualTransform):
```

