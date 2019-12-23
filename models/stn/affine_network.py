import torch
import torch.nn.functional as F
from torch import nn

from .layers import DownBlock


class AffineNetwork(nn.Module):
    def __init__(self):
        super(AffineNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ngf = 32
        # 288 x 384
        self.down1 = DownBlock(4, ngf, 3, 1, 1, True, 'leaky_relu', 'kaiming', False, True, False, True, True)
        # 144 x 192
        self.down2 = DownBlock(ngf, 2 * ngf, 3, 1, 1, True, 'leaky_relu', 'kaiming', False, True, False, True, True)
        # 72 x 96
        ngf *= 2
        self.down3 = DownBlock(ngf, 2 * ngf, 3, 1, 1, True, 'leaky_relu', 'kaiming', False, True, False, True, True)
        # 36 x 48
        self.down4 = DownBlock(ngf, 2 * ngf, 3, 1, 1, True, 'leaky_relu', 'kaiming', False, True, False, True, False)
        self.down5 = DownBlock(ngf, 2 * ngf, 3, 1, 1, True, 'leaky_relu', 'kaiming', False, True, False, True, False)
        self.convs = nn.Sequential(self.down1,
                                   self.down2,
                                   self.down3,
                                   self.down4,
                                   self.down5)
        self.local = nn.Sequential(nn.Linear(ngf * 36 * 84, 128, True),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Linear(128, 128, True),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Linear(128, 6, True))

        self.local[-1].weight.data.zero_()
        self.local[-1].bias.data.zero_()
        self.identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 128 * 36 * 84)
        dtheta = self.local(x)
        theta = dtheta + self.identity_theta
        grid = F.affine_grid(theta, x.size())
        return grid, dtheta
