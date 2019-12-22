"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage, RandomHorizontalFlip

import agrinetdata.ElbitRegisteration.util as elbit_util
from agrinetdata.ElbitRegisteration.Constants import FILE_NAMES
from agrinetdata.ElbitRegisteration.RegParam import RegParam
from data.base_dataset import BaseDataset


def make_dataset(dir, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    fid_list = []
    fid_to_A_path = {}
    fid_to_B_path = {}
    fid_found = []
    for root, _, fnames in sorted(os.walk(dir)):
        if root != dir:
            break
        for f in fnames:
            fid = f.split('_')[0]
            if fid not in fid_found:
                a = random.randint(0,2)
                b = random.randint(0,2)
                fid_list.append((fid,(a, b)))
                fid_found.append(fid)
            if 'RGB' in f:
                assert f not in fid_to_A_path
                fid_to_A_path[fid] = os.path.join(root, f)
            if 'Thermal' in f:
                assert f not in fid_to_B_path
                fid_to_B_path[fid] = os.path.join(root, f)

    random.shuffle(fid_list)
    return fid_list[:min(max_dataset_size, len(fid_list))], fid_to_A_path, fid_to_B_path


transform_to_tensor = ToTensor()
transform_to_pil = ToPILImage()


def crop_center(img, size=256):
    c, y, x = img.shape
    if not isinstance(size, tuple):
        startx = x // 2 - (size // 2)
        starty = y // 2 - (size // 2)
    return crop_image(img, (starty, startx), size)


def crop_image(img, pos, size):
    if isinstance(size, tuple):
        size_x = size[1]
        size_y = size[0]
    else:
        size_x = size_y = size
    if isinstance(pos, tuple):
        startx = pos[1]
        starty = pos[0]
    else:
        startx = starty = pos
    return img[..., starty:starty + size_y, startx:startx + size_x]


def transform_ir(img, crop=None, stride=128, crop_size=256):
    ir = img.astype(np.float32)
    # ir = cv2.resize(ir,(256,256))
    if (np.max(ir) - np.min(ir)) <= 1e-8:
        return None
    ir = 2 * ((ir - np.min(ir)) / (np.max(ir) - np.min(ir))) - 1
    if crop is not None:
        ir = cv2.resize(ir, dsize=None, fx=0.9, fy=0.9)
    ir = ir.reshape((1, *ir.shape))
    if crop is not None:
        c, h, w = ir.shape

        def get_pos(c, s):
            if c == 0:
                return 0
            elif c == 1:
                return s // 2 - (256 // 2)
            else:
                return s - 256

        pos_x = get_pos(crop[1], w)
        pos_y = get_pos(crop[0], h)
        ir = crop_center(ir,256) #crop_image(ir, (pos_y, pos_x), 256)

    ir = torch.from_numpy(ir)
    # ir = np.repeat(ir, 3, -1)
    return ir  # transform_to_tensor(ir)


def transform_rgb(img, crop=False):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)
    if (np.max(rgb) - np.min(rgb)) <= 1e-8:
        return None
    rgb = 2 * ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))) - 1
    if crop is not None:
        rgb = cv2.resize(rgb, dsize=None, fx=0.9, fy=0.9)
    rgb = np.rollaxis(rgb, -1, 0)
    # if crop:
    #     rgb = crop_center(rgb)
    if crop is not None:
        c, h, w = rgb.shape

        def get_pos(c, s):
            if c == 0:
                return 0
            elif c == 1:
                return s // 2 - (256 // 2)
            else:
                return s - 256

        pos_x = get_pos(crop[1], w)
        pos_y = get_pos(crop[0], h)
        rgb = crop_center(rgb, 256) # crop_image(rgb, (pos_y, pos_x), 256)
    rgb = torch.from_numpy(rgb)
    return rgb


def open_img_with_registeration_by_fid_new(frame_uri, fix_rgb=True, apply_depth_fix=True, use_elbit_eq=True):
    reg_param = RegParam().init_from_file('{}/{}'.format(frame_uri, 'Calib.txt'))
    rgb_img = cv2.imread('{}/{}'.format(frame_uri, FILE_NAMES['RGB']), cv2.IMREAD_UNCHANGED)
    thermal_img = cv2.imread('{}/{}'.format(frame_uri, FILE_NAMES['IR']), cv2.IMREAD_UNCHANGED)
    thermal_img = cv2.flip(thermal_img, 1)
    if apply_depth_fix:
        print('Trying to apply depth fix')
        depth_img = cv2.imread('{}/{}'.format(frame_uri, FILE_NAMES['RGBD']), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print('Couldn\'t apply depth fix')
        else:
            distance = elbit_util.calculate_distance(depth_img, use_elbit_eq=use_elbit_eq)
            reg_param.apply_dist_fix(distance)
    if fix_rgb:
        T = np.vstack((reg_param.T['IR'], np.array([[0, 0, 1], ])))
        T = np.linalg.inv(T)
        T = T[0:2, :]
        rgb_img = cv2.warpAffine(rgb_img, T,
                                 dsize=(thermal_img.shape[1], thermal_img.shape[0]))
    else:
        thermal_img = cv2.warpAffine(thermal_img, reg_param.T['IR'],
                                     dsize=(rgb_img.shape[1], rgb_img.shape[0]))
    return rgb_img, thermal_img


class AgriNetDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--apply_depth_fix', type=bool, default=True,
                            help='Apply depth fix on initial transformation')
        parser.add_argument('--use_elbit_eq', type=bool, default=False,
                            help='Use elbit equation for depth estimation')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.transform_ir = transform_ir
        self.transform_rgb = transform_rgb
        self.random_horizontal_flip = RandomHorizontalFlip(1.0)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.fid_list, self.fid_to_A_path, self.fid_to_B_path = make_dataset(self.dir_AB, opt.max_dataset_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.fix_rgb = opt.fix_rgb
        self.apply_depth_fix = opt.apply_depth_fix
        self.use_elbit_eq = opt.use_elbit_eq
        self._crop = True

    def __getitem__(self, index):
        """Return a data point and its metadata information.

                Parameters:
                    index - - a random integer for data indexing

                Returns a dictionary that contains A, B, A_paths and B_paths
                    A (tensor) - - an image in the input domain
                    B (tensor) - - its corresponding image in the target domain
                    A_paths (str) - - image paths
                    B_paths (str) - - image paths (same as A_paths)
                """
        # read a image given a random integer index
        fid, crop = self.fid_list[index]
        crop = (random.randint(0,2), random.randint(0,2))
        A_path = self.fid_to_A_path[fid]
        B_path = self.fid_to_B_path[fid]
        # A, B = open_img_with_registeration_by_fid_new(AB_path, self.fix_rgb, self.apply_depth_fix, self.use_elbit_eq)
        # apply the same transform to both A and B
        flip = random.randint(0, 1)
        A, B = cv2.imread(A_path, cv2.IMREAD_UNCHANGED), cv2.imread(B_path, cv2.IMREAD_UNCHANGED)
        if flip == 1:
            A = cv2.flip(A, 1)
            B = cv2.flip(B, 1)
        A = transform_rgb(A, crop=None)
        B = transform_ir(B, crop=None)
        if A is None:
            print('Bad image {}'.format(A_path))
            return []
        if B is None:
            print('Bad image {}'.format(B_path))
            return []
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.fid_list)
