import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .affine_network import AffineNetwork
from .unet import UnetSTN
from .unet_config import UNETConfig


class STN(nn.Module):
    def __init__(self, in_channels_a, in_channels_b, height, width, batch_size, cfg='A'):
        super(STN, self).__init__()
        self.cnt = 0
        self.batch_size = batch_size
        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
        self.offset_map = UnetSTN(self.in_channels_a + self.in_channels_b, cfg=UNETConfig(cfg=cfg)).to(self.device)
        self.affine_map = AffineNetwork().to(self.device)
        self.identity_grid = self.get_identity_grid()

    def get_scale(self):
        scale_x = np.ones((self.batch_size, self.oh, self.ow)).astype(np.float32) * (256.0 / float(self.ow))
        scale_y = np.ones((self.batch_size, self.oh, self.ow)).astype(np.float32) * (256.0 / float(self.oh))
        scale = np.stack([scale_x, scale_y], axis=1)
        return torch.from_numpy(scale)

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def apply_offset(self, img, offset):
        b_size = img.size(0)
        grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + offset).permute([0, 2, 3, 1])
        ret = F.grid_sample(img, grid)
        return ret

    def forward(self, img_a, img_b, apply_on=None):
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)
        # Perform Affine Transformation
        img_conc = torch.cat((img_a, img_b), 1)
        theta, dtheta = self.affine_map(img_conc)
        affine_offset = F.affine_grid(theta, (self.oh, self.ow))
        img_a_t = self.apply_offset(img_a, affine_offset)
        img_conc = torch.cat((img_a_t, img_b), 1)
        deformable_offset = self.offset_map(img_conc)
        offset = deformable_offset + affine_offset
        if apply_on is None:
            x = self.apply_offset(img_a, offset)
        else:
            x = self.apply_offset(apply_on, offset)
        return x, offset

    def init_to_identity(self):
        self.offset_map.init_to_identity()
