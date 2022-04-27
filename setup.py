try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='volumentations_3D',
    version='1.0.4',
    author='Roman Sol (ZFTurbo), ashawkey, qubvel, muellerdo',
    packages=['volumentations', 'volumentations/augmentations', 'volumentations/core'],
    url='https://github.com/ZFTurbo/volumentations',
    description='Library for 3D augmentations',
    long_description='Library for 3D augmentations. Inspired by albumentations.'
                     'More details: https://github.com/ZFTurbo/volumentations',
    install_requires=[
        'scikit-image',
        'scipy',
        'opencv-python',
        "numpy",
    ],
)
