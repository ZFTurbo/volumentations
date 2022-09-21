# Volumentations 3D

3D Volume data augmentation package inspired by albumentations.

Volumentations is a working project, which originated from the following Git repositories:
- Original:                 https://github.com/albumentations-team/albumentations
- 3D Conversion:            https://github.com/ashawkey/volumentations
- Continued Development:    https://github.com/ZFTurbo/volumentations

Nevertheless, if you are using this subpackage, please give credit to all authors including ashawkey, ZFTurbo, qubvel and muellerdo.

Initially inspired by [albumentations](https://github.com/albumentations-team/albumentations) library for augmentation of 2D images.

# Installation

```sh
pip install volumentations-3D
```

### Simple Example

```python
from volumentations import *

def get_augmentation(patch_size):
    return Compose([
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        RandomCropFromBorders(crop_value=0.1, p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.2),
        RandomGamma(gamma_limit=(80, 120), p=0.2),
    ], p=1.0)

aug = get_augmentation((64, 128, 128))

img = np.random.randint(0, 255, size=(128, 256, 256), dtype=np.uint8)
lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

# with mask
data = {'image': img, 'mask': lbl}
aug_data = aug(**data)
img, lbl = aug_data['image'], aug_data['mask']

# without mask
data = {'image': img}
aug_data = aug(**data)
img = aug_data['image']

```

* Check working usage example in [tst_volumentations_type_1.py](tst_volumentations_type_1.py)  
* Added another usage example / testing in [tst_volumentations_type_2.py](tst_volumentations_type_2.py)  

# Difference from initial version

* Diverse bug fixes.
* Implemented multiple augmentations.
* Approximation enhancements to be closer to Albumentations.

### Implemented 3D augmentations

```python
PadIfNeeded
GaussianNoise
Resize
RandomScale
RotatePseudo2D
RandomRotate90
Flip
Normalize
Float
Contiguous
Transpose
CenterCrop
RandomResizedCrop
RandomCrop
CropNonEmptyMaskIfExists
ResizedCropNonEmptyMaskIfExists
RandomGamma
ElasticTransformPseudo2D
ElasticTransform
Rotate
RandomCropFromBorders
GridDropout
RandomDropPlane
RandomBrightnessContrast
ColorJitter
```

### Speed table

Speed in seconds per one sample.

| Aug name | Cube = 64px | Cube = 96px | Cube = 128px | Cube = 224px | Cube = 256px |
|----------|-------------|-------------|--------------|--------------|--------------|
| Rotate | 0.0402 | 0.1366 | 0.3246 | 1.7546 | 2.6349 | 
| RandomCropFromBorders| 0.0037 | 0.0129 | 0.0315 | 0.1634 | 0.2426 |
| ElasticTransform | 0.1588 | 0.5439 | 2.8649 | 11.8937 | 42.3886 |
| Resize (type = 0) | 0.4029 | 0.4077 | 0.4245 | 0.5545 | 0.6278 |
| Resize (type = 1) | 0.3618 | 0.3696 | 0.3871 | 0.5174 | 0.5896 |
| Flip | 0.0042 | 0.0134 | 0.0314 | 0.1649 | 0.2453 |
| RandomRotate90 | 0.0040 | 0.0140 | 0.0306 | 0.1672 | 0.2439 |
| GaussianNoise | 0.0143 | 0.0406 | 0.0956 | 0.4992 | 0.7381 |
| RandomGamma | 0.0066 | 0.0211 | 0.0505 | 0.2654 |  0.3989 |
| RandomScale | 0.0158 | 0.0518 | 0.1198 | 0.6391 | 0.9457 |

## Citation

For more details, please refer to the publication: https://doi.org/10.1016/j.compbiomed.2021.105089

If you find this code useful, please cite it as:
```
@article{solovyev20223d,
  title={3D convolutional neural networks for stalled brain capillary detection},
  author={Solovyev, Roman and Kalinin, Alexandr A and Gabruseva, Tatiana},
  journal={Computers in Biology and Medicine},
  volume={141},
  pages={105089},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2021.105089}
}
```
