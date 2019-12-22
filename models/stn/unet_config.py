down_nf = {'A': [64, 128, 128, 128, 128, 128, 128],
           'B': [64, 128, 128, 128, 128, 128, 128],
           'C': [64, 64, 128, 128, 128, 128, 128],
           'D': [32, 32, 64, 64, 64, 64, 64],
           'E': [32, 64, 64, 64, 64, 64, 64]}

up_nf = {'A': [128, 128, 128, 128, 128, 128, 64],
         'B': [128, 128, 128, 128, 128, 128, 64],
         'C': [128, 128, 128, 128, 128, 64, 64],
         'D': [64, 64, 64, 64, 64, 32, 32],
         'E': [64, 64, 64, 64, 64, 64, 32]}

output_refine_nf = {'A': [64, 64],
                    'B': [64, 32],
                    'C': [64, 64],
                    'D': [32, 32],
                    'E': [32, 32]}
refine = {'A': False,
          'B': False,
          'C': False,
          'D': False,
          'E': False}

refine_input = {'A': False,
                'B': False,
                'C': False,
                'D': False,
                'E': False}

use_residual_block = {'A': False,
                      'B': False,
                      'C': False,
                      'D': False,
                      'E': False}

init_type = {'A': 'kaiming',
             'B': 'kaiming',
             'C': 'kaiming',
             'D': 'kaiming',
             'E': 'kaiming'}


class UNETConfig:
    def __init__(self, cfg='A'):
        self.down_nf = down_nf[cfg]
        self.down_activation = 'leaky_relu'
        self.up_nf = up_nf[cfg]
        self.up_activation = 'leaky_relu'
        self.output_refine_nf = output_refine_nf[cfg]
        self.output_refine_activation = 'leaky_relu'
        self.refine = refine[cfg]
        self.refine_input = refine_input[cfg]
        self.init_type = init_type[cfg]
        self.use_norm = False
        self.use_bias = True
        self.refine_input = refine_input[cfg]
