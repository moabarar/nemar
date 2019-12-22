from functools import partial

import torch
from torch import nn

from ..networks import ResnetBlock

scale_eval = False

alpha = 0.02
beta = 0.00002

resnet_n_blocks = 1


def custom_init(m):
    m.data.normal_(0.0, alpha)


def get_init_function(activation, init_function, **kwargs):
    a = 0.0
    if activation is 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function is 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        elif init_function is 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function is 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function is 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function is 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0,
                           std=1e-4)  # partial(torch.nn.init.uniform_, a=-2e-2, b=2e-2)#  # torch.nn.init.zeros_
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function


def get_activation(activation, **kwargs):
    if activation is 'relu':
        return nn.ReLU(inplace=True)
    elif activation is 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    elif activation is 'tanh':
        return nn.Tanh()
    elif activation is 'sigmoid':
        return nn.Sigmoid()
    else:
        return None


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='relu',
                 init_func=None, use_norm=False, use_resnet=False, **kwargs):
        super(Conv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformer(out_channels, resnet_n_blocks, init_func,
                                              use_bias=not use_norm) if use_resnet else None
        self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False) if use_norm else None
        # self.norm = nn.BatchNorm2d(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        # Initialize the weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv2d.weight)
        if self.conv2d.bias is not None:
            self.conv2d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 1.0, 0.02)
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='relu',
                 init_func=None, use_norm=False, refine=True, upsample=True, use_resnet=False, **kwargs):
        super(UpBlock, self).__init__()
        alpha = int(in_channels // out_channels)
        intermediate_nfeats = 2 * in_channels // alpha
        self.conv_0 = Conv(in_channels, intermediate_nfeats, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=bias, activation=activation, init_func=init_func, use_norm=use_norm,
                           use_resnet=use_resnet, **kwargs)
        self.conv_1 = None
        if refine:
            self.conv_1 = Conv(intermediate_nfeats, intermediate_nfeats, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, bias=bias, activation=activation, init_func=init_func,
                               use_norm=use_norm, use_resnet=use_resnet, **kwargs)
        self.upsample = upsample
        if self.upsample:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.up_conv = Conv(intermediate_nfeats, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding,
                                bias=bias, activation=None, init_func=init_func, use_norm=use_norm, use_resnet=False,
                                **kwargs)

    def forward(self, x):
        x = self.conv_0(x)
        if self.conv_1 is not None:
            x = self.conv_1(x)
        if self.upsample:
            x = self.up(x)
            x = self.up_conv(x)
        return x


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='relu',
                 init_func=None, use_norm=False, use_resnet=False, skip=True, refine=False, pool=True, **kwargs):
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
            self.pool = nn.MaxPool2d(kernel_size=2)

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


class ResnetTransformer(torch.nn.Module):
    def __init__(self, dim, n_blocks, init_func, use_bias=False, norm='instance'):
        super(ResnetTransformer, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False,
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
