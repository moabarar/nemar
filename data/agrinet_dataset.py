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
from torchvision.transforms import ToTensor, ToPILImage, RandomHorizontalFlip, Normalize

from data.base_dataset import BaseDataset


rgb_unit_normalize = Normalize(mean=[118.6808, 118.4811, 85.3022],
                      std=[60.1607, 63.4259, 57.6958])

def make_dataset(dir, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    fid_list = []
    fid_to_A_path = {}
    fid_to_B_path = {}
    for root, _, fnames in sorted(os.walk(dir)):
        if root != dir:
            break
        for f in fnames:
            fid = f.split('_')[0]
            if fid not in fid_list:
                fid_list.append(fid)
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


def transform_ir(img, unit_normalize=False):
    ir = img.astype(np.float32)
    if unit_normalize:
        ir = (ir - 2260.7612)/161.9801
    else:
        if (np.max(ir) - np.min(ir)) <= 1e-8:
           return None
        ir = 2.0 * ((ir - np.min(ir)) / (np.max(ir) - np.min(ir))) - 1.0 
        # ir = 2.0*(ir-735.)/(5720. - 735.) - 1.0
    ir = ir.reshape((1, *ir.shape))
    ir = torch.from_numpy(ir)
    return ir


def transform_rgb(img, unit_normalize=False):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)
    if not unit_normalize:
        if (np.max(rgb) - np.min(rgb)) <= 1e-8:
            return None
        rgb = 2.0*(rgb/255.0) - 1.0
        # for ch in range(rgb.shape[-1]):
        #     mx = rgb[...,ch].max()
        #     mn = rgb[...,ch].min()
        #     rgb[...,ch] -= mn
        #     rgb[...,ch] *= 2.0/(mx-mn)
        #     rgb[...,ch] -= 1.0
    rgb = np.rollaxis(rgb, -1, 0)
    rgb = torch.from_numpy(rgb)
    if unit_normalize:
        rgb = rgb_unit_normalize(rgb)
    return rgb

def random_roi(ih, iw, oh, ow):
    left_most = random.randint(0,iw-ow-1) if iw != ow else 0
    upper_most = random.randint(0,ih-oh-1) if ih != oh else 0
    return left_most, upper_most

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
        self._crop  = True
        self._crop = (self.opt.img_height != 288 or self.opt.img_width != 384)
        self.ih, self.iw = self.opt.img_height, self.opt.img_width

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
        fid = self.fid_list[index]
        A_path = self.fid_to_A_path[fid]
        B_path = self.fid_to_B_path[fid]
        A, B = cv2.imread(A_path, cv2.IMREAD_UNCHANGED), cv2.imread(B_path, cv2.IMREAD_UNCHANGED)
        if self._crop:
            left, up = random_roi(288, 384, self.opt.img_height, self.opt.img_width)
            A = A[up:up+self.ih, left:left+self.iw,...]
            B = B[up:up+self.ih, left:left+self.iw]
        # if self.resize:
        #     A = cv2.resize(A,(self.opt.img_width, self.opt.img_height))
        #     B = cv2.resize(B,(self.opt.img_width, self.opt.img_height))
        flip = np.random.randint(0,2)
        if flip != 0:
            A = cv2.flip(A, 1)
            B = cv2.flip(B, 1)
        A = transform_rgb(A)
        B = transform_ir(B) 
           
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
