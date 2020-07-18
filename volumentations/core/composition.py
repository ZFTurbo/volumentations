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
