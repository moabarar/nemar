import itertools

import torch

from util.losses import smoothness_loss, l2_norm
from util.tb_visualizer import TensorboardVisualizer
from . import networks
from .base_model import BaseModel
from .stn.stn import STN


def reg_loss(deformation):
    return torch.mean(deformation ** 2)


class MIRNETModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we int:roduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_G', type=float, default=1.0, help='weight for loss (A -> B)')
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--alpha_reg', type=float, default=0.0, help='Biliteral alpha')
            parser.add_argument('--stn_lr', type=float, default=0.0,
                                help='Learning rate used for optimizing the stn model.')
            parser.add_argument('--stn_lr_high', type=float, default=0.0,
                                help='High learning rate used for optimizing the stn model.')
            parser.add_argument('--lambda_stn_reg', type=float, default=0.0, help='Regulization term for STN')
            parser.add_argument('--lambda_edge_loss', type=float, default=0.0, help='Edge loss lambda')
            parser.add_argument('--stn_cfg', type=str, default='A', help='STN configuration')
            parser.add_argument('--tbvis_enable', action='store_true', help='Enable tensorboard visualizer')
            parser.add_argument('--stn_train_with_gan', action='store_true',
                                help='whether to train stn with gan')
            parser.add_argument('--stn_train_with_edge', action='store_true',
                                help='whether to train stn with edge loss')
            parser.add_argument('--stn_train_with_l1', action='store_true',
                                help='whether to train stn the generator with l1')
            parser.add_argument('--stn_condition_on_discriminator', action='store_true',
                                help='whether to condition the stn on the discriminator')
            TensorboardVisualizer.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # Setup the visualizers
        self.train_stn = True
        self.setup_visualizers()
        if self.isTrain and opt.tbvis_enable:
            self.tb_visualizer = TensorboardVisualizer(self, ['netSTN_A'], self.loss_names, self.opt)
        else:
            self.tb_visualizer = None
        self.define_networks()
        if self.tb_visualizer is not None:
            print('Enabling Tensorboard Visualizer!')
            self.tb_visualizer.enable()
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            self.setup_optimizers()

    def setup_visualizers(self):
        loss_names_A = ['L1_B', 'GAN_B', 'L1_P', 'GAN_P', 'smoothness', 'D_fake_P', 'D_fake_B', 'D']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['fake_B_B', 'fake_B_P', 'transformed_real_A', 'fake_B']

        model_names_a = ['G_A', 'STN_A']
        if self.isTrain:
            model_names_a += ['D_A']

        self.visual_names = ['real_A', 'real_B']
        self.model_names = []
        self.loss_names = []
        # if self.opt.direction == 'AtoB':
        self.visual_names += visual_names_A
        self.model_names += model_names_a
        self.loss_names += loss_names_A

    def define_networks(self):
        opt = self.opt
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        AtoB = opt.direction == 'AtoB'
        in_c = opt.input_nc if AtoB else opt.output_nc
        out_c = opt.output_nc if AtoB else opt.input_nc
        # if opt.direction in ['AtoB', 'Cycle']:
        self.netG_A = networks.define_G(in_c, out_c, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netSTN_A = STN(in_channels_a=in_c, in_channels_b=out_c, height=opt.img_height, width=opt.img_width,
                            batch_size=opt.batch_size, cfg=opt.stn_cfg)
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netSTN_A.to(self.gpu_ids[0])
            self.netSTN_A = torch.nn.DataParallel(self.netSTN_A, self.gpu_ids)  # multi-GPUs
            self.netSTN_A.module.init_to_identity()
        else:
            self.netSTN_A.init_to_identity()
        if self.isTrain:  # define discriminator
            self.netD_A = networks.define_D(opt.output_nc + opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

    def reset_weights(self):
        opt = self.opt
        # if opt.direction in ['AtoB', 'Cycle']:
        networks.init_weights(self.netG_A, opt.init_type, opt.init_gain)
        networks.init_weights(self.netD_A, opt.init_type, opt.init_gain)

    def setup_optimizers(self):
        self.optimizer_STN_A, self.optimizer_G_A, self.optimizer_D_A = self.get_GAN_A_optimizer()
        self.optimizers.append(self.optimizer_G_A)
        self.optimizers.append(self.optimizer_D_A)
        self.optimizers.append(self.optimizer_STN_A)

    def get_GAN_A_optimizer(self):
        opt = self.opt
        return (
            torch.optim.Adam(itertools.chain(self.netSTN_A.parameters()),
                             lr=opt.stn_lr, betas=(opt.beta1, 0.999)),
            torch.optim.Adam([{'params': self.netG_A.parameters(), 'betas': (opt.beta1, 0.999),
                               'lr': opt.lr}]),
            torch.optim.Adam(itertools.chain(self.netD_A.parameters()),
                             lr=opt.lr, betas=(opt.beta1, 0.999))

        )

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB' or self.opt.direction == 'Cycle'
        if AtoB:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']
        else:
            self.real_A = input['B'].to(self.device)
            self.real_B = input['A'].to(self.device)
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # A -> B
        # if self.opt.direction in ['AtoB', 'Cycle']:
        # Before
        self.transformed_real_A, self.transformation_grid, self.deformation_field, self.affine_theta = self.netSTN_A(
            self.real_A, self.real_B)
        self.fake_B_B = self.netG_A(self.transformed_real_A)

        # Parallel
        self.fake_B = self.netG_A(self.real_A)
        self.fake_B_P = self.netSTN_A.module.apply_grid(self.fake_B, self.transformation_grid)

    def backward_G_A(self):
        """Calculate GAN and L1 loss for the generator"""
        # STN Before:
        # ----> Reconstruction loss:
        self.loss_L1_B = self.opt.lambda_L1 * self.criterionL1(self.fake_B_B, self.real_B)
        # ----> GAN loss:
        fake_AB_t = torch.cat((self.real_A, self.fake_B_B), 1)
        pred_fake = self.netD_A(fake_AB_t)
        self.loss_GAN_B = self.opt.lambda_G * self.criterionGAN(pred_fake, True)

        # STN Parallel:
        # ----> Reconstruction loss:
        self.loss_L1_P = self.opt.lambda_L1 * self.criterionL1(self.fake_B_P, self.real_B)
        # ----> GAN loss:
        fake_AB_t = torch.cat((self.real_A, self.fake_B_P), 1)
        pred_fake = self.netD_A(fake_AB_t)
        self.loss_GAN_P = self.opt.lambda_G * self.criterionGAN(pred_fake, True)

        # STN Regularization:
        self.loss_smoothness = self.opt.lambda_stn_reg * smoothness_loss(self.deformation_field,
                                                                         img=self.transformed_real_A.detach(),
                                                                         alpha=self.opt.alpha_reg)
        self.loss_affine_reg = 1.0 * l2_norm(self.affine_theta)

        loss = self.loss_L1_B + self.loss_L1_P + self.loss_GAN_B + self.loss_GAN_P + self.loss_smoothness + self.affine_theta
        loss.backward()

        return loss

    def backward_D_A(self):
        """Calculate GAN loss for the discriminator"""
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD_A(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)

        # STN Before:
        # ----> Fake
        fake_AB = torch.cat((self.real_A, self.fake_B_B), 1)
        pred_fake = self.netD_A(fake_AB.detach())
        self.loss_D_fake_B = self.criterionGAN(pred_fake, False)

        # STN Parallel:
        # ----> Parallel
        fake_AB = torch.cat((self.real_A, self.fake_B_P), 1)
        pred_fake = self.netD_A(fake_AB.detach())
        self.loss_D_fake_P = self.criterionGAN(pred_fake, False)

        # combine loss and calculate gradients
        self.loss_D = 0.5 * (loss_D_real + self.loss_D_fake_B + self.loss_D_fake_P)
        self.loss_D.backward()

        return self.loss_D

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # Optimize GAN A
        # if self.opt.direction in ['AtoB']:
        # Backward D_A
        self.set_requires_grad([self.netG_A, self.netSTN_A], False)
        self.optimizer_D_A.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.optimizer_D_A.step()  # update D_A and D_B's weights
        self.set_requires_grad([self.netG_A, self.netSTN_A], True)

        # Backward D_B
        self.set_requires_grad([self.netD_A], False)
        self.optimizer_STN_A.zero_grad()
        self.optimizer_G_A.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G_A()  # calculate gradients for G_A and G_B
        self.optimizer_STN_A.step()
        self.optimizer_G_A.step()
        self.set_requires_grad([self.netD_A], True)
        # else:
        #     exit(-1)

        if self.tb_visualizer is not None:
            self.tb_visualizer.iteration_step()
