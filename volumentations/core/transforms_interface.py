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
import random
import numpy as np
from typing import Sequence, Tuple

# DEBUG only flag
VERBOSE = False


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


class Transform:
    def __init__(self, always_apply=False, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.always_apply = always_apply

    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            if VERBOSE:
                print('RUN', self.__class__.__name__, params)

            for k, v in data.items():
                if k in targets[0]:
                    data[k] = self.apply(v, **params)
                else:
                    data[k] = v

        return data

    def get_params(self, **data):
        """
        shared parameters for one apply. (usually random values)
        """
        return {}

    def apply(self, volume, **params):
        raise NotImplementedError


class DualTransform(Transform):
    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            if VERBOSE:
                print('RUN', self.__class__.__name__, params)

            for k, v in data.items():
                if k in targets[0]:
                    data[k] = self.apply(v, **params)
                elif k in targets[1]:
                    data[k] = self.apply_to_mask(v, **params)
                else:
                    data[k] = v

        return data

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)


class ImageOnlyTransform(Transform):
    """Transform applied to image only."""

    @property
    def targets(self):
        return {"image": self.apply}
