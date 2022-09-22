# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from volumentations import *
import time


def tst_volumentations_speed():
    total_volumes_to_check = 100
    sizes_list = [
        (64, 64, 64),
        (96, 96, 96),
        (128, 128, 128),
        (224, 224, 224),
        (256, 256, 256),
    ]

    for size in sizes_list:
        patch_size1 = (32, 32, 32)
        patch_size2 = (200, 200, 200)

        full_list_to_check = [
            Rotate((-15, 15), (-15, 15), (-15, 15), p=1.0),
            RandomCropFromBorders(crop_value=0.1, p=1.0),
            ElasticTransform((0, 0.25), interpolation=2, p=1.0),
            Resize(patch_size1, interpolation=1, resize_type=0, always_apply=True, p=1.0),
            Resize(patch_size1, interpolation=1, resize_type=1, always_apply=True, p=1.0),
            Resize(patch_size2, interpolation=1, resize_type=0, always_apply=True, p=1.0),
            Resize(patch_size2, interpolation=1, resize_type=1, always_apply=True, p=1.0),
            Flip(0, p=1.0),
            Flip(1, p=1.0),
            Flip(2, p=1.0),
            RandomRotate90((1, 2), p=1.0),
            GaussianNoise(var_limit=(0, 5), p=1.0),
            RandomGamma(gamma_limit=(80, 120), p=1.0),
            RandomScale(scale_limit=[0.9, 1.1], interpolation=1, always_apply=True, p=1.0)
        ]

        for f in full_list_to_check:
            name = f.__class__.__name__
            aug1 = Compose([
                f,
            ], p=1.0)

            start_time = time.time()
            data = []
            for i in range(total_volumes_to_check):
                data.append(np.random.uniform(low=0.0, high=255, size=size))

            for i, cube in enumerate(data):
                try:
                    cube1 = aug1(image=cube)['image']
                except Exception as e:
                    print('Augmentation error: {}'.format(str(e)))
                    continue

            delta = time.time() - start_time
            print('Size: {} Aug: {} Time: {:.2f} sec Per sample: {:.4f} sec'.format(size, name, delta, delta / len(data)))
            print(f.__dict__)


if __name__ == '__main__':
    tst_volumentations_speed()
