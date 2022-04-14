# Volumentations 3D

Volumentations is a subpackage of AUCMEDI, which originated from the following Git repositories:
- Original:                 https://github.com/albumentations-team/albumentations
- 3D Conversion:            https://github.com/ashawkey/volumentations
- Continued Development:    https://github.com/ZFTurbo/volumentations
- Enhancements:             https://github.com/qubvel/volumentations

Due to a stop of ongoing development in this subpackage, we decided to create a new repository.

Nevertheless, if you are using this subpackage, please give credit to all authors including ashawkey, ZFTurbo and qubvel.

Initially inspired by [albumentations](https://github.com/albumentations-team/albumentations) library for augmentation of 2D images.

# Installation

```sh
# Original
pip install volumentations-3D

# Fork
pip install volumentations-aucmedi
```

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

# with mask
data = {'image': img, 'mask': lbl}
aug_data = aug(**data)
img, lbl = aug_data['image'], aug_data['mask']

# without mask
data = {'image': img}
aug_data = aug(**data)
img = aug_data['image']

```

Check working usage example in [tst_volumentations.py](tst_volumentations.py)

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

## Credits and License

Added some credits/license to each file.

```
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
```
