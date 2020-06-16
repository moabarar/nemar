from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from ..networks import ResnetBlock

scale_eval = False

alpha = 0.02
beta = 0.00002

resnet_n_blocks = 1

norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
align_corners = False
up_sample_mode = 'bilinear'


def custom_init(m):
    m.data.normal_(0.0, alpha)


def get_init_function(activation, init_function, **kwargs):
    """Get the initialization function from the given name."""
    a = 0.0
    if activation == 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function == 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation, mode='fan_in')
        elif init_function == 'dirac':
            return torch.nn.init.dirac_
        elif init_function == 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function


def get_activation(activation, **kwargs):
    """Get the appropriate activation from the given name"""
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return None


class Conv(torch.nn.Module):
    """Defines a basic convolution layer.
    The general structure is as follow:

    Conv -> Norm (optional) -> Activation -----------> + --> Output
                                         |            ^
                                         |__ResBlcok__| (optional)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, **kwargs):
        super(Conv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformer(out_channels, resnet_n_blocks, init_func) if use_resnet else None
        self.norm = norm_layer(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        # Initialize the weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv2d.weight)
        if self.conv2d.bias is not None:
            self.conv2d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x


class UpBlock(torch.nn.Module):
    def __init__(self, nc_down_stream, nc_skip_stream, nc_out, kernel_size, stride, padding, bias=True,
                 activation='relu',
                 init_func='kaiming', use_norm=False, refine=False, use_resnet=False, use_add=False,
                 use_attention=False, **kwargs):
        super(UpBlock, self).__init__()
        if 'nc_inner' in kwargs:
            nc_inner = kwargs['nc_inner']
        else:
            nc_inner = nc_out
        self.conv_0 = Conv(nc_down_stream + nc_skip_stream, nc_inner, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=bias, activation=activation, init_func=init_func, use_norm=use_norm,
                           use_resnet=use_resnet, **kwargs)
        self.conv_1 = None
        if refine:
            self.conv_1 = Conv(nc_inner, nc_inner, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                               activation=activation, init_func=init_func, use_norm=use_norm, use_resnet=use_resnet,
                               **kwargs)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention_gate = AttentionGate(nc_down_stream, nc_skip_stream, nc_inner, use_norm=True,
                                                init_func=init_func)
        self.up_conv = Conv(nc_inner, nc_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                            activation=activation, init_func=init_func, use_norm=use_norm, use_resnet=False, **kwargs)
        self.use_add = use_add
        if self.use_add:
            self.output = Conv(nc_out, 2, kernel_size=1, stride=1, padding=0, bias=bias, activation=None,
                               init_func='zeros',
                               use_norm=False, use_resnet=False)

    def forward(self, down_stream, skip_stream):
        down_stream_size = down_stream.size()
        skip_stream_size = skip_stream.size()
        if self.use_attention:
            skip_stream = self.attention_gate(down_stream, skip_stream)
        if down_stream_size[2] != skip_stream_size[2] or down_stream_size[3] != skip_stream_size[3]:
            down_stream = F.interpolate(down_stream, (skip_stream_size[2], skip_stream_size[3]),
                                        mode=up_sample_mode, align_corners=align_corners)
        x = torch.cat([down_stream, skip_stream], 1)
        x = self.conv_0(x)
        if self.conv_1 is not None:
            x = self.conv_1(x)
        if self.use_add:
            x = self.output(x) + down_stream
        else:
            x = self.up_conv(x)
        return x


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, skip=True, refine=False, pool=True,
                 pool_size=2, **kwargs):
        super(DownBlock, self).__init__()
        self.conv_0 = Conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                           activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                           use_resnet=use_resnet, **kwargs)
        self.conv_1 = None
        if refine:
            self.conv_1 = Conv(out_channels, out_channels, kernel_size, stride, padding, bias=bias,
                               activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                               use_resnet=use_resnet, **kwargs)
        self.skip = skip
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = skip = self.conv_0(x)
        if self.conv_1 is not None:
            x = skip = self.conv_1(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.skip:
            return x, skip
        else:
            return x


class AttentionGate(torch.nn.Module):
    def __init__(self, nc_g, nc_x, nc_inner, use_norm=False, init_func='kaiming', mask_channel_wise=False):
        super(AttentionGate, self).__init__()
        self.conv_g = Conv(nc_g, nc_inner, 1, 1, 0, bias=True, activation=None, init_func=init_func,
                           use_norm=use_norm, use_resnet=False)
        self.conv_x = Conv(nc_x, nc_inner, 1, 1, 0, bias=False, activation=None, init_func=init_func,
                           use_norm=use_norm, use_resnet=False)
        self.residual = nn.ReLU(inplace=True)
        self.mask_channel_wise = mask_channel_wise
        self.attention_map = Conv(nc_inner, nc_x if mask_channel_wise else 1, 1, 1, 0, bias=True, activation='sigmoid',
                                  init_function=init_func, use_norm=use_norm, use_resnet=False)

    def forward(self, g, x):
        x_size = x.size()
        g_size = g.size()
        x_resized = x
        g_c = self.conv_g(g)
        x_c = self.conv_x(x_resized)
        if x_c.size(2) != g_size[2] and x_c.size(3) != g_size[3]:
            x_c = F.interpolate(x_c, (g_size[2], g_size[3]), mode=up_sample_mode, align_corners=align_corners)
        combined = self.residual(g_c + x_c)
        alpha = self.attention_map(combined)
        if not self.mask_channel_wise:
            alpha = alpha.repeat(1, x_size[1], 1, 1)
        alpha_size = alpha.size()
        if alpha_size[2] != x_size[2] and alpha_size[3] != x_size[3]:
            alpha = F.interpolate(x, (x_size[2], x_size[3]), mode=up_sample_mode, align_corners=align_corners)
        return alpha * x


class ResnetTransformer(torch.nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(dim, padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                            use_bias=True)]
        self.model = nn.Sequential(*model)

        init_ = get_init_function('relu', init_func)

        def init_weights(m):
            if type(m) == nn.Conv2d:
                init_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if type(m) == nn.BatchNorm2d:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
