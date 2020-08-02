import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DownBlock, Conv, ResnetTransformer
from .stn_losses import smoothness_loss

sampling_align_corners = False
sampling_mode = 'bilinear'

# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 64, 64, 64, 64, 64, 64], }
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {'A': [64, 64, 64, 64, 64, 64, 32], }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 3, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }


class ResUnet(torch.nn.Module):
    """Predicts a dense deofmration field that aligns two given images.

    The networks is unet-based network with (possibly) residual blocks. The residual blocks may be used in the
    in the down-sampling path, on the encoded features and prior to the deformation field generation."""
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks[cfg], use_norm=False))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks[cfg]:
            self.c1 = Conv(in_nf, 2 * in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks[cfg] == 0
                      else ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func))
            self.c2 = Conv(2 * in_nf, in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(self, 'up_{}'.format(conv_num),
                    Conv(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, bias=True, activation=act,
                         init_fun=init_func, use_norm=False, use_resnet=False))
            in_nf = out_nf
            conv_num -= 1
        if refine_output[cfg]:
            self.refine = nn.Sequential(ResnetTransformer(in_nf, 1, init_func),
                                        Conv(in_nf, in_nf, 1, 1, 0, use_resnet=False, init_func=init_func,
                                             activation=act,
                                             use_norm=False)
                                        )

        else:
            self.refine = lambda x: x
        self.output = Conv(in_nf, 2, 3, 1, 1, use_resnet=False, bias=True,
                           init_func=('zeros' if init_to_identity else init_func), activation=None,
                           use_norm=False)

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear')
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x


class UnetSTN(nn.Module):
    """This class is generates and applies the deformable transformation on the input images."""

    def __init__(self, in_channels_a, in_channels_b, height, width, cfg, init_func, stn_bilateral_alpha,
                 init_to_identity, multi_resolution_regularization):
        super(UnetSTN, self).__init__()
        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offset_map = ResUnet(self.in_channels_a, self.in_channels_b, cfg, init_func, init_to_identity).to(
            self.device)
        self.identity_grid = self.get_identity_grid()
        self.alpha = stn_bilateral_alpha
        self.multi_resolution_regularization = multi_resolution_regularization

    def get_identity_grid(self):
        """Returns a sampling-grid that represents the identity transformation."""
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def get_grid(self, img_a, img_b, return_offsets_only=False):
        """Return the predicted sampling grid that aligns img_a with img_b."""
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)
        # Get Deformation Field
        b_size = img_a.size(0)
        deformation = self.offset_map(img_a, img_b)
        deformation_upsampled = deformation
        if deformation.size(2) != self.oh and deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(deformation, (self.oh, self.ow), mode=sampling_mode,
                                                  align_corners=sampling_align_corners)
        if return_offsets_only:
            resampling_grid = deformation_upsampled.permute([0, 2, 3, 1])
        else:
            resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])
        return resampling_grid

    def forward(self, img_a, img_b, apply_on=None):
        """
        Predicts the spatial alignment needed to align img_a with img_b. The spatial transformation will be applied
        on the tensors passed by apply_on (if apply_on is None then the transformation will be applied on img_a).

            :param img_a: the source image.
            :param img_b: the target image.
            :param apply_on: the geometric transformation can be applied on different tensors provided by this list.
                        If not set, then the transformation will be applied on img_a.
            :return: a list of the warped images (matching the order they appeared in apply on), and the regularization term
                        calculated for the predicted transformation."""
        if img_a.is_cuda and not self.identity_grid.is_cuda:
            self.identity_grid = self.identity_grid.to(img_a.device)
        # Get Deformation Field
        b_size = img_a.size(0)
        deformation = self.offset_map(img_a, img_b)
        deformation_upsampled = deformation
        if deformation.size(2) != self.oh and deformation.size(3) != self.ow:
            deformation_upsampled = F.interpolate(deformation, (self.oh, self.ow), mode=sampling_mode)
        resampling_grid = (self.identity_grid.repeat(b_size, 1, 1, 1) + deformation_upsampled).permute([0, 2, 3, 1])
        # Wrap image wrt to the defroamtion field
        if apply_on is None:
            apply_on = [img_a]
        warped_images = []
        for img in apply_on:
            warped_images.append(F.grid_sample(img, resampling_grid, mode=sampling_mode, padding_mode='zeros',
                                               align_corners=sampling_align_corners))
        # Calculate STN regulization term
        reg_term = self._calculate_regularization_term(deformation, warped_images[0])
        return warped_images, reg_term

    def _calculate_regularization_term(self, deformation, img):
        """Calculate the regularization term of the predicted deformation.
        The regularization may-be applied to different resolution for larger images."""
        dh, dw = deformation.size(2), deformation.size(3)
        img = None if img is None else img.detach()
        reg = 0.0
        factor = 1.0
        for i in range(self.multi_resolution_regularization):
            if i != 0:
                deformation_resized = F.interpolate(deformation, (dh // (2 ** i), dw // (2 ** i)), mode=sampling_mode,
                                                    align_corners=sampling_align_corners)
                img_resized = F.interpolate(img, (dh // (2 ** i), dw // (2 ** i)), mode=sampling_mode,
                                            align_corners=sampling_align_corners)
            elif deformation.size()[2::] != img.size()[2::]:
                deformation_resized = deformation
                img_resized = F.interpolate(img, deformation.size()[2::], mode=sampling_mode,
                                            align_corners=sampling_align_corners)
            else:
                deformation_resized = deformation
                img_resized = img
            reg += factor * smoothness_loss(deformation_resized, img_resized, alpha=self.alpha)
            factor /= 2.0
        return reg
