from setuptools import setup
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
   name='volumentations-aucmedi',
   version='1.0.1',
   description='Library for 3D augmentations. Inspired by albumentations.',
   url='https://github.com/muellerdo/volumentations',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='MIT',
   long_description=long_description,
   long_description_content_type="text/markdown",
   packages=find_packages(),
   python_requires='>=3.6',
   install_requires=['numpy',
                     'scikit-image',
                     'scipy',
                     'opencv-python'],
   classifiers=["Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Operating System :: OS Independent",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Recognition"]
)
