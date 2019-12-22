import os.path
import random
from data.base_dataset import  BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import h5py
import numpy as np
import torch
import joblib
import math

from skimage.transform import warp, AffineTransform


def get_affine_params(angle, center=None, new_center=None, scale=None):
    angle = -angle / 180.0 * math.pi
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = nx + new_center[0], ny + new_center[1]
    if scale:
        (sx, sy) = scale

    cosine = math.cos(angle)
    sine = math.sin(angle)

    a = cosine * sx
    b = sine
    c = - sine
    d = cosine * sy

    translate_x = nx - x
    translate_y = ny - y
    return a, b, c, d, translate_x, translate_y


def depth_mapping(xy, depth_image, affine_params, use_depth=True):
    results = []
    depth_image = depth_image - depth_image.min()
    depth_image = depth_image / depth_image.max()
    depth_vector = 1 - depth_image[xy[:, 1].astype(np.uint8), xy[:, 0].astype(np.uint8)]
    depth_vector[depth_vector == 0] = depth_vector[depth_vector > 0].min()  # 0 causes hard artifacts

    for ind in range(len(xy)):
        x, y = xy[ind, :]

        depth = depth_vector[ind]

        if not use_depth:
            depth = 1

        # if depth == 0:
        #     a = 5
            # results.append(np.array([x, y]))
            # continue

        angle = 0
        scale = 1
        translation = [0,0]
        angle = affine_params[0] * depth
        scale = 1 - affine_params[1] * depth
        translation = [affine_params[2] * depth, affine_params[3] * depth]

        cosine = math.cos(angle)
        sine = math.sin(angle)

        new_x = scale * cosine * x + sine * x + translation[0]
        new_y = scale * cosine * y - sine * x + translation[1]

        results.append(np.array([new_x, new_y]))

    results = np.array(results)
    return results


def overlay_depth(img, depth):

    depth = (depth - depth.min())/depth.max()
    depth = depth/depth.max()
    depth *= 255
    overlay = np.reshape(img, (depth.shape[0] * depth.shape[1], 3))
    overlay[:, 2] = overlay[:, 1]
    overlay[:, 1] = depth[:].reshape(depth.shape[0] * depth.shape[1])
    overlay = overlay.reshape(depth.shape[0], depth.shape[1], 3).astype(np.uint8)
    return overlay


def show_array(array):
    Image.fromarray(array.astype(np.uint8)).show()

A_path = '/mnt/data/yiftach/aligning_cyclegan/datasets/nyu_depth/trainA/0.jpg'
B_path = '/mnt/data/yiftach/aligning_cyclegan/datasets/nyu_depth/trainB/0.jpg'
A_img = np.array(Image.open(A_path).convert('RGB'))
B_img = np.array(Image.open(B_path).convert('L'))

max_angle=40
max_scale=0.5
max_translation=10
angle = np.random.uniform(-max_angle, max_angle)
# angle = -angle / 180.0 * math.pi
angle=math.radians(angle)
scale = np.random.uniform(-max_scale, max_scale)
# scale = 1
center_x, center_y = (np.random.uniform(-max_translation, max_translation), np.random.uniform(-max_translation, max_translation))
# center_x = center_y = 0
# affine_params = get_affine_params(angle, [a // 2 for a in B_img.shape], (center_x, center_y))

warp_args = {'depth_image': B_img, 'affine_params': (angle, scale, center_x, center_y)}
warp_args1 = {'depth_image': B_img, 'affine_params': (angle, scale, center_x, center_y), 'use_depth':False}

warped = warp(A_img, depth_mapping, map_args=warp_args, output_shape=A_img.shape)
warped1 = warp(A_img, depth_mapping, map_args=warp_args1, output_shape=A_img.shape)
a = 5