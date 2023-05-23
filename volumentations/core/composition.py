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
from ..augmentations import transforms as T


class Compose:
    def __init__(self, transforms, p=1.0, targets=[['image'],['mask']]):
        assert 0 <= p <= 1
        self.transforms = [T.Float(always_apply=True)] + transforms + [T.Contiguous(always_apply=True)]
        self.p = p
        self.targets = targets

    def get_always_apply_transforms(self):
        res = []
        for tr in self.transforms:
            if tr.always_apply:
                res.append(tr)
        return res

    def __call__(self, force_apply=False, **data):
        need_to_run = force_apply or random.random() < self.p
        transforms = self.transforms if need_to_run else self.get_always_apply_transforms()

        for tr in transforms:
            data = tr(force_apply, self.targets, **data)

        return data


class ComposeChoice:
    def __init__(self, transforms, p=1.0, n=1, targets=[['image'], ['mask']]):
        assert 0 <= p <= 1
        self.transforms = transforms
        self.p = p
        self.n = n
        self.targets = targets

    def __call__(self, **data):
        if random.random() > self.p:
            return data

        transforms = random.sample(self.transforms, self.n)
        transforms = [T.Float()] + transforms + [T.Contiguous()]

        for tr in transforms:
            data = tr(True, self.targets, **data)

        return data
