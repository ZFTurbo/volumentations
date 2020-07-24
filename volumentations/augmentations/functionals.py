import numpy as np
import skimage.transform as skt
import scipy.ndimage.interpolation as sci
from scipy.ndimage import zoom
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
#import SimpleITK as sitk
from warnings import warn

"""
vol: [H, W, D(, C)]

x, y, z <--> H, W, D

you should give (H, W, D) form shape.

skimage interpolation notations:

order = 0: Nearest-Neighbor
order = 1: Bi-Linear (default)
order = 2: Bi-Quadratic 
order = 3: Bi-Cubic
order = 4: Bi-Quartic
order = 5: Bi-Quintic

Interpolation behaves strangely when input of type int.
** Be sure to change volume and mask data type to float !!! **
"""


def rotate2d(img, angle, axes=(0,1), reshape=False, interpolation=1, border_mode='constant', value=0):
    return sci.rotate(img, angle, axes, reshape=reshape, order=interpolation, mode=border_mode, cval=value)


def shift(img, shift, interpolation=1, border_mode='constant', value=0):
    return sci.shift(img, shift, order=interpolation, mode=border_mode, cval=value)


def crop(img, x1, y1, z1, x2, y2, z2):
    height, width, depth = img.shape[:3]
    if x2 <= x1 or y2 <= y1 or z2 <= z1:
        raise ValueError
    if x1 < 0 or y1 < 0 or z1 < 0:
        raise ValueError
    if x2 > height or y2 > width or z2 > depth:
        img = pad(img, (x2, y2, z2))
        warn('image size smaller than crop size, pad by default.', UserWarning)

    return img[x1:x2, y1:y2, z1:z2]


def get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth):
    x1 = (height - crop_height) // 2
    x2 = x1 + crop_height
    y1 = (width - crop_width) // 2
    y2 = y1 + crop_width
    z1 = (depth - crop_depth) // 2
    z2 = z1 + crop_depth
    return x1, y1, z1, x2, y2, z2


def center_crop(img, crop_height, crop_width, crop_depth):
    height, width, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError
    x1, y1, z1, x2, y2, z2 = get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
    img = img[x1:x2, y1:y2, z1:z2]
    return img


def get_random_crop_coords(height, width, depth, crop_height, crop_width, crop_depth, h_start, w_start, d_start):
    x1 = int((height - crop_height) * h_start)
    x2 = x1 + crop_height
    y1 = int((width - crop_width) * w_start)
    y2 = y1 + crop_width
    z1 = int((depth - crop_depth) * d_start)
    z2 = z1 + crop_depth
    return x1, y1, z1, x2, y2, z2


def random_crop(img, crop_height, crop_width, crop_depth, h_start, w_start, d_start):
    height, width, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        img = pad(img, (crop_width, crop_height, crop_depth))
        warn('image size smaller than crop size, pad by default.', UserWarning)
    x1, y1, z1, x2, y2, z2 = get_random_crop_coords(height, width, depth, crop_height, crop_width, crop_depth, h_start, w_start, d_start)
    img = img[x1:x2, y1:y2, z1:z2]
    return img


def normalize(img, range_norm=True):
    if range_norm:
        mn = img.min()
        mx = img.max()
        img = (img - mn) / (mx - mn)
    mean = img.mean()
    std = img.std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return img


def pad(image, new_shape, border_mode="constant", value=0):
    '''
    image: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    '''
    axes_not_pad = len(image.shape) - len(new_shape)

    old_shape = np.array(image.shape[:len(new_shape)])
    new_shape = np.array([max(new_shape[i], old_shape[i]) for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    pad_list = [list(i) for i in zip(pad_below, pad_above)] + [[0, 0]] * axes_not_pad

    if border_mode == 'reflect':
        res = np.pad(image, pad_list, border_mode)
    elif border_mode == 'constant':
        res = np.pad(image, pad_list, border_mode, constant_values=value)
    else:
        raise ValueError

    return res


def gaussian_noise(img, var, mean):
    return img + np.random.normal(mean, var, img.shape)


def resize(img, new_shape, interpolation=1):
    """
    img: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    """
    type = 1

    if type == 0:
        new_img = skt.resize(img, new_shape, order=interpolation, mode='constant', cval=0, clip=True, anti_aliasing=False)
    else:
        shp = tuple(np.array(new_shape) / np.array(img.shape[:3]))
        # Multichannel
        data = []
        for i in range(img.shape[-1]):
            d0 = zoom(img[..., i].astype(np.uint8).copy(), shp, order=interpolation)
            data.append(d0.copy())
        new_img = np.stack(data, axis=3)

    return new_img


def rescale(img, scale, interpolation=1):
    """
    img: [H, W, D, C] or [H, W, D]
    scale: scalar float
    """
    return skt.rescale(img, scale, order=interpolation, mode='constant', cval=0, clip=True, multichannel=True, anti_aliasing=False)
    """
    shape = [int(scale * i) for i in img.shape[:3]]
    return resize(img, shape, interpolation)
    """


def gamma_transform(img, gamma, eps=1e-7):
    mn = img.min()
    rng = img.max() - mn
    img = (img - mn)/(rng + eps)
    return 255 * np.power(img, gamma)


def elastic_transform_pseudo2D(img, alpha, sigma, alpha_affine, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None, random_state=42, approximate=False):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    random_state = np.random.RandomState(random_state)

    height, width, depth = img.shape[:3]

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    # pseoudo 2D
    res = np.zeros_like(img)
    for d in range(depth):
        tmp = img[:, :, d] # [H, W, C]
        tmp = cv2.warpAffine(tmp, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)
        res[:, :, d] = tmp
    img = res


    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha

        dy = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
        dy *= alpha
    else:
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # pseoudo 2D
    res = np.zeros_like(img)
    for d in range(depth):
        tmp = img[:, :, d] # [H, W, C]
        tmp = cv2.remap(tmp, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value)
        res[:, :, d] = tmp
    img = res

    return img


"""
Later are coordinates-based 3D rotation and elastic transforms.
reference: https://github.com/MIC-DKFZ/batchgenerators
"""

def elastic_transform(img, sigmas, alphas, interpolation=1, border_mode='constant', value=0, random_state=42):
    """
    img: [H, W, D(, C)]
    """
    coords = generate_coords(img.shape[:3])
    coords = elastic_deform_coords(coords, sigmas, alphas, random_state)
    coords = recenter_coords(coords)
    if len(img.shape) == 4:
        num_channels = img.shape[3]
        res = []
        for channel in range(num_channels):
            res.append(map_coordinates(img[:,:,:,channel], coords, order=interpolation, mode=border_mode, cval=value))
        return np.stack(res, -1)
    else:
        return map_coordinates(img, coords, order=interpolation, mode=border_mode, cval=value)


def generate_coords(shape):
    """
    coords: [n_dim=3, H, W, D]
    """
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2)[d]
    return coords


def elastic_deform_coords(coords, sigmas, alphas, random_state):
    random_state = np.random.RandomState(random_state)
    n_dim = coords.shape[0]
    if not isinstance(alphas, (tuple, list)):
        alphas = [alphas] * n_dim
    if not isinstance(sigmas, (tuple, list)):
        sigmas = [sigmas] * n_dim
    offsets = []
    for d in range(n_dim):
        offset = gaussian_filter((random_state.rand(*coords.shape[1:]) * 2 - 1), sigmas, mode="constant", cval=0)
        mx = np.max(np.abs(offset))
        offset = alphas[d] * offset / mx
        offsets.append(offset)
    offsets = np.array(offsets)
    coords += offsets
    return coords


def recenter_coords(coords):
    n_dim = coords.shape[0]
    mean = coords.mean(axis=tuple(range(1, len(coords.shape))), keepdims=True)
    coords -= mean
    for d in range(n_dim):
        ctr = int(np.round(coords.shape[d+1]/2))
        coords[d] += ctr
    return coords


def rotate3d(img, x, y, z, interpolation=1, border_mode='constant', value=0):
    """
    img: [H, W, D(, C)]
    x, y, z: angle in degree.
    """
    x, y, z = [np.pi*i/180 for i in [x, y, z]]
    coords = generate_coords(img.shape[:3])
    coords = rotate_coords(coords, x, y, z)
    coords = recenter_coords(coords)
    if len(img.shape) == 4:
        num_channels = img.shape[3]
        res = []
        for channel in range(num_channels):
            res.append(map_coordinates(img[:,:,:,channel], coords, order=interpolation, mode=border_mode, cval=value))
        return np.stack(res, -1)
    else:
        return map_coordinates(img, coords, order=interpolation, mode=border_mode, cval=value)


def rotate_coords(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = rot_matrix @ rot_x(angle_x)
    rot_matrix = rot_matrix @ rot_y(angle_y)
    rot_matrix = rot_matrix @ rot_z(angle_z)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def rot_x(angle):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    return rotation_x


def rot_y(angle):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    return rotation_y


def rot_z(angle):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    return rotation_z


def rescale_warp(img, scale, interpolation=1):
    """
    img: [H, W, D(, C)]
    """
    coords = generate_coords(img.shape[:3])
    coords = scale_coords(coords, scale)
    coords = recenter_coords(coords)
    if len(img.shape) == 4:
        num_channels = img.shape[3]
        res = []
        for channel in range(num_channels):
            res.append(map_coordinates(img[:,:,:,channel], coords, order=interpolation, mode=border_mode, cval=value))
        return np.stack(res, -1)
    else:
        return map_coordinates(img, coords, order=interpolation, mode=border_mode, cval=value)


def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords


def clamping_crop(img, sh0_min, sh1_min, sh2_min, sh0_max, sh1_max, sh2_max):
    d, h, w = img.shape[:3]
    if sh0_min < 0:
        sh0_min = 0
    if sh1_min < 0:
        sh1_min = 0
    if sh2_min < 0:
        sh2_min = 0
    if sh0_max > d:
        sh0_max = d
    if sh1_max > h:
        sh1_max = h
    if sh2_max > w:
        sh2_max = w
    return img[int(sh0_min): int(sh0_max), int(sh1_min): int(sh1_max), int(sh2_min): int(sh2_max)]


def cutout(img, holes, fill_value=0):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.copy()
    for x1, y1, z1, x2, y2, z2 in holes:
        img[y1:y2, x1:x2, z1:z2] = fill_value
    return img
