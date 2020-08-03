# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from volumentations import *
import os
import cv2
import urllib.request


OUTPUT_DIR = './debug_videos/'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def read_video(f):
    cap = cv2.VideoCapture(f)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    frame_list = []
    print('ID: {} Video length: {} Width: {} Height: {} FPS: {}'.format(os.path.basename(f), length, width, height, fps))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame.copy())
        current_frame += 1

    frame_list = np.array(frame_list, dtype=np.uint8)
    return frame_list


def get_augmentation(patch_size):
    return Compose([
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        RandomCropFromBorders(crop_value=0.1, p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        RandomDropPlane(plane_drop_prob=0.1, axes=(0, 1, 2), p=0.5),
        Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.5),
        RandomGamma(gamma_limit=(0.5, 1.5), p=0.5),
    ], p=1.0)


def create_video(image_list, out_file, fps):
    height, width = image_list[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = -1
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)
    for im in image_list:
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=2)
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def tst_volumentations():
    number_of_aug_videos = 10
    out_shape = (150, 224, 360)
    inp_video = 'sample.mp4'
    if not os.path.isfile(inp_video):
        print('Downloading sample.mp4...')
        urllib.request.urlretrieve('https://github.com/ZFTurbo/volumentations/releases/download/v1.0/sample.mp4', inp_video)

    cube = read_video(inp_video)
    print('Sample video shape: {}'.format(cube.shape))
    aug = get_augmentation(out_shape)
    for i in range(number_of_aug_videos):
        print('Aug: {}'.format(i))
        data = {'image': cube}
        aug_data = aug(**data)
        img = aug_data['image']
        create_video(img, OUTPUT_DIR + 'video_test_{}.avi'.format(i), 24)


if __name__ == '__main__':
    tst_volumentations()
