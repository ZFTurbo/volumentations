import pytest
import numpy as np

from volumentations import *


augmentations = [
    CenterCrop,
    ColorJitter,
    Contiguous,
    # CropNonEmptyMaskIfExists,
    Downscale,
    ElasticTransform,
    ElasticTransformPseudo2D,
    Flip,
    Float,
    GaussianNoise,
    GlassBlur,
    GridDistortion,
    GridDropout,
    ImageCompression,
    Normalize,
    PadIfNeeded,
    RandomBrightnessContrast,
    RandomCrop,
    RandomCropFromBorders,
    RandomDropPlane,
    RandomGamma,
    RandomResizedCrop,
    RandomRotate90,
    RandomScale,
    RandomScale2,
    RemoveEmptyBorder,
    Resize,
    # ResizedCropNonEmptyMaskIfExists,
    Rotate,
    RotatePseudo2D,
    Transpose,
]

arg_required = [
    CenterCrop,
    CropNonEmptyMaskIfExists,
    PadIfNeeded,
    RandomCrop,
    RandomResizedCrop,
    Resize,
    ResizedCropNonEmptyMaskIfExists,
]


@pytest.fixture(scope="module")
def cube():
    X, Y, Z = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
    values = np.sin(X*Y*Z) / (X*Y*Z)
    return values


@pytest.mark.parametrize("aug_class", augmentations)
def test_augmentations(aug_class, cube):
    if aug_class in arg_required:
        aug = aug_class(shape=(30, 30, 30))
    else:
        aug = aug_class()
    new_cube = aug(True, "image", image=cube)["image"]
    print(aug_class.__name__, new_cube.shape)
