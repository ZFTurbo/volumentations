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
import numpy as np
from typing import Optional, Sequence, Union, Type, Any
import random as py_random

NumType = Union[int, float, np.ndarray]
IntNumType = Union[int, np.ndarray]
Size = Union[int, Sequence[int]]

def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(py_random.randint(0, (1 << 32) - 1))

def randint(
    low: IntNumType,
    high: Optional[IntNumType] = None,
    size: Optional[Size] = None,
    dtype: Type = np.int32,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randint(low, high, size, dtype)

def uniform(
    low: NumType = 0.0,
    high: NumType = 1.0,
    size: Optional[Size] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.uniform(low, high, size)

def normal(
    loc: NumType = 0.0,
    scale: NumType = 1.0,
    size: Optional[Size] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.normal(loc, scale, size)
