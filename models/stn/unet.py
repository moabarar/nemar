import torch
from torch import nn
from torch.nn import functional as F

from .layers import Conv, DownBlock, UpBlock
from .unet_config import UNETConfig


class UnetSTN(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, cfg=UNETConfig()):
        super(UnetSTN, self).__init__()
        # Initializer UNET Architecture:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        down_activation = cfg.down_activation
        up_activation = cfg.up_activation
        output_refine_activation = cfg.output_refine_activation
        init_function = cfg.init_type
        refine = cfg.refine
        prev_input_nf = input_nc
        use_norm = cfg.use_norm
        use_bias = cfg.use_bias
        skip_connect = []
        use_resnet = cfg.use_resnet
        ksize = 3
        padding = (ksize - 1) // 2
        for i, nf in enumerate(cfg.down_nf):
            setattr(self, 'down_{}'.format(i + 1), DownBlock(prev_input_nf, nf, kernel_size=ksize, stride=1,
                                                             padding=padding, bias=use_bias, activation=down_activation,
                                                             init_func=init_function, use_resnet=use_resnet,
                                                             use_norm=use_norm, pool=True,
                                                             refine=refine or (i == 0 and cfg.refine_input)))
            prev_input_nf = nf
            skip_connect.insert(0, 'down_{}'.format(i + 1))
        skip_connect.insert(0, None)
        self.transform_block = DownBlock(prev_input_nf, prev_input_nf, kernel_size=ksize, stride=1, padding=padding,
                                         bias=use_bias, activation=down_activation, init_func=init_function,
                                         use_resnet=use_resnet, use_norm=use_norm, pool=False, refine=refine,
                                         skip=False)
        connection_map = {}
        for i, nf in enumerate(cfg.up_nf):
            setattr(self, 'up_{}'.format(i + 1),
                    UpBlock(prev_input_nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias,
                            activation=up_activation, init_func=init_function, use_norm=use_norm, refine=refine,
                            use_resnet=use_resnet))
            if i < len(skip_connect) and skip_connect[i] != None:
                connection_map['up_{}'.format(i + 1)] = skip_connect[i]
            if i + 1 < len(skip_connect) and skip_connect[i + 1] != None:
                prev_input_nf = nf + cfg.down_nf[-i - 1]
            else:
                prev_input_nf = nf
        for i, nf in enumerate(cfg.output_refine_nf):
            kernel_size = 1 if i == len(cfg.output_refine_nf) - 1 else 3
            padding = (kernel_size - 1) // 2
            setattr(self, 'output_refine_{}'.format(i + 1),
                    Conv(prev_input_nf, nf, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias,
                         activation=output_refine_activation, init_func=init_function, use_norm=use_norm,
                         pool=False, skip=False, use_resnet=use_resnet))
            if i == 0 and len(connection_map.keys()) < len(skip_connect) - 1:
                connection_map['output_refine_{}'.format(i + 1)] = skip_connect[-1]
            prev_input_nf = nf if isinstance(nf, int) else prev_input_nf
        self.output = Conv(prev_input_nf, 2, kernel_size=3, stride=1, padding=1,
                           bias=True, activation=None, init_func='zeros', use_norm=False, pool=False)
        self.connection_map = connection_map
        self.skip_connect = skip_connect

    def _pad(self, x, y):
        pad_h = y.size(2) - x.size(2)
        pad_w = y.size(3) - x.size(3)
        pad_h_1 = pad_h // 2
        pad_h_2 = pad_h - pad_h_1
        pad_w_1 = pad_w // 2
        pad_w_2 = pad_w - pad_w_1
        pad = (pad_w_1, pad_w_2, pad_h_1, pad_h_2)
        if all(p == 0 for p in pad):
            return x

        return F.pad(x, pad, 'constant', 0.0)

    def forward(self, x):
        skip_vals = {}
        i = 1
        while hasattr(self, 'down_{}'.format(i)):
            l_name = 'down_{}'.format(i)
            tmp = getattr(self, l_name)(x)
            if l_name in self.skip_connect:
                skip_vals[l_name] = tmp[1]
                x = tmp[0]
            else:
                x = tmp
            i += 1
        x = self.transform_block(x)
        i = 1
        while hasattr(self, 'up_{}'.format(i)):
            l_name = 'up_{}'.format(i)
            if l_name in self.connection_map:
                x = self._pad(x, skip_vals[self.connection_map[l_name]])
                x = torch.cat([x, skip_vals[self.connection_map[l_name]]],
                              1)  # x + skip_vals[self.connection_map[l_name]]
            x = getattr(self, l_name)(x)
            i += 1
        i = 1
        while hasattr(self, 'output_refine_{}'.format(i)):
            l_name = 'output_refine_{}'.format(i)
            if l_name in self.connection_map:
                x = self._pad(x, skip_vals[self.connection_map[l_name]])
                x = torch.cat([x, skip_vals[self.connection_map[l_name]]],
                              1)  # x + skip_vals[self.connection_map[l_name]]
            x = getattr(self, l_name)(x)
            i += 1
        x = self.output(x)
        return x

    def init_to_identity(self):
        return
        # self.output.conv2d.weight.data.normal_(0.0,1e-5)
        # self.output.conv2d.bias.data.zero_()
