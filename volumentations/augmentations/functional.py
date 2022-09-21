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
from functools import wraps
import skimage.transform as skt
import scipy.ndimage.interpolation as sci
from scipy.ndimage import zoom
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
from warnings import warn

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


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


def preserve_shape(func):
    """
    Preserve shape of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function

def rotate2d(img, angle, axes=(0,1), reshape=False, interpolation=1, border_mode='reflect', value=0):
    return sci.rotate(img, angle, axes, reshape=reshape, order=interpolation, mode=border_mode, cval=value)


def shift(img, shift, interpolation=1, border_mode='reflect', value=0):
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
    else:
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
    if np.isinf(denominator).any():
        img[...] = 0
    else:
        img = (img - mean) * denominator
    return img


def pad(image, new_shape, border_mode="reflect", value=0):
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


def gaussian_noise(img, gauss):
    img = img.astype("float32")
    return img + gauss


def resize(img, new_shape, interpolation=1, resize_type=0):
    """
    img: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    interpolation: The order of the spline interpolation (0-5)
    resize_type: what type of resize to use: scikit-image or
    """

    if resize_type == 0:
        new_img = skt.resize(
            img,
            new_shape,
            order=interpolation,
            mode='reflect',
            cval=0,
            clip=True,
            anti_aliasing=False
        )
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
    return skt.rescale(img, scale, order=interpolation, mode='reflect', cval=0,
                       clip=True, channel_axis=-1, anti_aliasing=False)
    """
    shape = [int(scale * i) for i in img.shape[:3]]
    return resize(img, shape, interpolation)
    """


@preserve_shape
def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)

    return img


def elastic_transform_pseudo2D(img, alpha, sigma, alpha_affine, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None, random_state=42, approximate=False):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    random_state = np.random.RandomState(random_state)

    depth, height, width  = img.shape[:3]

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
        tmp = img[d, :, :] # [D, H, W, C]
        tmp = cv2.warpAffine(tmp, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)
        res[d, :, :] = tmp
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

def elastic_transform(img, sigmas, alphas, interpolation=1, border_mode='reflect', value=0, random_state=42):
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


def rotate3d(img, x, y, z, interpolation=1, border_mode='reflect', value=0):
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

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function

@clipped
def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img

@preserve_shape
def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += beta * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img

def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


def _adjust_brightness_torchvision_uint8(img, factor):
    lut = np.arange(0, 256) * factor
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)

@preserve_shape
def adjust_brightness_torchvision(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    if img.dtype == np.uint8:
        return _adjust_brightness_torchvision_uint8(img, factor)

    return clip(img * factor, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])

def _adjust_contrast_torchvision_uint8(img, factor, mean):
    lut = np.arange(0, 256) * factor
    lut = lut + mean * (1 - factor)
    lut = clip(lut, img.dtype, 255)

    return cv2.LUT(img, lut)

@preserve_shape
def adjust_contrast_torchvision(img, factor):
    if factor == 1:
        return img

    if is_2Dgrayscale_image(img):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        return np.full_like(img, int(mean + 0.5), dtype=img.dtype)

    if img.dtype == np.uint8:
        return _adjust_contrast_torchvision_uint8(img, factor, mean)

    return clip(
        img.astype(np.float32) * factor + mean * (1 - factor),
        img.dtype,
        MAX_VALUES_BY_DTYPE[img.dtype],
    )


@preserve_shape
def adjust_saturation_torchvision(img, factor, gamma=0):
    if factor == 1:
        return img

    if is_2Dgrayscale_image(img):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    if img.dtype == np.uint8:
        return result

    # OpenCV does not clip values for float dtype
    return clip(result, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_hue_torchvision_uint8(img, factor):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = cv2.LUT(img[..., 0], lut)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue_torchvision(img, factor):
    if is_2Dgrayscale_image(img):
        return img

    if factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_torchvision_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def is_3Drgb_image(image):
    return len(image.shape) == 4 and image.shape[-1] == 3

def is_3Dgrayscale_image(image):
    return (len(image.shape) == 3) or (len(image.shape) == 4 and image.shape[-1] == 1)

def is_2Drgb_image(image):
    return len(image.shape) == 3 and image.shape[-1] == 3

def is_2Dgrayscale_image(image):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """
    def get_num_channels(image):
        return image.shape[2] if len(image.shape) == 3 else 1

    @wraps(process_fn)
    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn

@preserve_shape
def grid_distortion(
    img,
    num_steps=10,
    xsteps=(),
    ysteps=(),
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
):
    """Perform a grid distortion of an input image.
    Reference:
        http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)

@preserve_shape
def downscale(img, scale, interpolation=cv2.INTER_NEAREST):
    shape_org = img.shape[:3]
    shape_down = tuple([int(x*scale) for x in shape_org])

    need_cast = interpolation != cv2.INTER_NEAREST and img.dtype == np.uint8
    if need_cast:
        img = to_float(img)



    downscaled = skt.resize(img, shape_down, order=interpolation, mode='reflect',
                            cval=0, clip=True, anti_aliasing=False)
    upscaled = skt.resize(downscaled, shape_org, order=interpolation, mode='reflect',
                            cval=0, clip=True, anti_aliasing=False)
    if need_cast:
        upscaled = from_float(np.clip(upscaled, 0, 1), dtype=np.dtype("uint8"))
    return upscaled

def glass_blur(img, sigma, max_delta, iterations, dxy, mode):
    x = cv2.GaussianBlur(np.array(img), sigmaX=sigma, ksize=(0, 0))

    if mode == "fast":

        hs = np.arange(img.shape[0] - max_delta, max_delta, -1)
        ws = np.arange(img.shape[1] - max_delta, max_delta, -1)
        h = np.tile(hs, ws.shape[0])
        w = np.repeat(ws, hs.shape[0])

        for i in range(iterations):
            dy = dxy[:, i, 0]
            dx = dxy[:, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]

    elif mode == "exact":
        for ind, (i, h, w) in enumerate(
            product(
                range(iterations),
                range(img.shape[0] - max_delta, max_delta, -1),
                range(img.shape[1] - max_delta, max_delta, -1),
            )
        ):
            ind = ind if ind < len(dxy) else ind % len(dxy)
            dy = dxy[ind, i, 0]
            dx = dxy[ind, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]

    return cv2.GaussianBlur(x, sigmaX=sigma, ksize=(0, 0))

@preserve_shape
def image_compression(img, quality, image_type):
    if image_type in [".jpeg", ".jpg"]:
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        warn(
            "Image compression augmentation "
            "is most effective with uint8 inputs, "
            "{} is used as input.".format(input_dtype),
            UserWarning,
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for image augmentation".format(input_dtype))

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    if needs_float:
        img = to_float(img, max_value=255)
    return img

def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).astype(dtype)

def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(img.dtype)
            )
    return img.astype("float32") / max_value
