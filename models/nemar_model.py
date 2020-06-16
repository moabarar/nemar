import itertools

import torch
import torch.nn.functional as F

import models.stn as stn
from util.tb_visualizer import TensorboardVisualizer
from . import networks
from .base_model import BaseModel


class NEMARModel(BaseModel):
    """
    NeMAR: a neural multimodal adversarial image registration network.
    This class train a registration network and a geometry preserving translation network network. This is done
    using three networks:

    netT - A translation network that translates from modality A --to--> modality B (by default a
    netR - A registration network that applies geometric transformation to spatially align modality A --with--> modality B
    netD - Adversarial network that discriminates between fake an real images.

    Official implementation of:
    Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation paper
    https://arxiv.org/abs/2003.08073

    Inspired by the implementation of pix2pix:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Modify the command line."""
        parser = stn.modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Weight for the GAN loss.')
            parser.add_argument('--lambda_recon', type=float, default=100.0,
                                help='Weight for the L1 reconstruction loss.')
            parser.add_argument('--lambda_smooth', type=float, default=0.0, help='Regularization term used by the STN')
            parser.add_argument('--enable_tbvis', action='store_true',
                                help='Enable tensorboard visualizer (default : False)')
            parser.add_argument('--multi_resolution', type=int, default=1,
                                help='Use of multi-resolution discriminator.'
                                     '(if equals to 1 then no multi-resolution training is applied)')
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
        if self.isTrain and opt.enable_tbvis:
            self.tb_visualizer = TensorboardVisualizer(self, ['netR', 'netT', 'netD'], self.loss_names, self.opt)
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
        # <Loss>_TR denotes the loss for the translation first variant.
        # <Loss>_RT denotes the loss for the registration first variant.
        loss_names_A = ['L1_TR', 'GAN_TR', 'L1_RT', 'GAN_RT', 'smoothness', 'D_fake_TR', 'D_fake_RT', 'D']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['fake_TR_B', 'fake_RT_B', 'registered_real_A', 'fake_B']

        model_names_a = ['T', 'R']
        if self.isTrain:
            model_names_a += ['D']

        self.visual_names = ['real_A', 'real_B']
        self.model_names = []
        self.loss_names = []
        # if self.opt.direction == 'AtoB':
        self.visual_names += visual_names_A
        self.model_names += model_names_a
        self.loss_names += loss_names_A

    def define_networks(self):
        # define networks:
        # netT - is the photometric translation network (i.e the generator)
        # netR - is the registration network (i.e STN)
        # netD - is the discriminator network
        opt = self.opt
        # Support two directions (A->B) or (B->A)
        AtoB = opt.direction == 'AtoB'
        in_c = opt.input_nc if AtoB else opt.output_nc
        out_c = opt.output_nc if AtoB else opt.input_nc
        self.netT = networks.define_G(in_c, out_c, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netR = stn.define_stn(self.opt, self.opt.stn_type)
        if self.isTrain:  # define discriminator
            self.netD = networks.define_D(opt.output_nc + opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # We support multi-resolution discriminator - this could yield better performance for large input images.
            self.netD_multiresolution = []
            if opt.multi_resolution > 1:
                for _ in range(self.opt.multi_resolution - 1):
                    netD_S = networks.define_D(opt.output_nc + opt.input_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                    self.netD_multiresolution.append(netD_S)

    def reset_weights(self):
        # We have tested what happens if we reset the discriminator/translation network's weights during training.
        # This eventually will results in th
        opt = self.opt
        networks.init_weights(self.netT, opt.init_type, opt.init_gain)
        networks.init_weights(self.netD, opt.init_type, opt.init_gain)
        for netD_S in self.netD_multiresolution:
            networks.init_weights(netD_S, opt.init_type, opt.init_gain)

    def setup_optimizers(self):
        opt = self.opt

        # Define optimizer for the registration network:
        self.optimizer_R = torch.optim.Adam(itertools.chain(self.netR.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999), )
        # Define optimizer for the translation network:
        self.optimizer_T = torch.optim.Adam([{'params': self.netT.parameters(), 'betas': (opt.beta1, 0.999),
                                              'lr': opt.lr}])
        # Define optimizer for the discriminator network:
        d_params = self.netD.parameters()
        if opt.multi_resolution > 1:
            d_params = itertools.chain(d_params, *[x.parameters() for x in self.netD_multiresolution])
        self.optimizer_D = torch.optim.Adam(d_params, lr=opt.lr, betas=(opt.beta1, 0.999))

        self.optimizers.append(self.optimizer_T)
        self.optimizers.append(self.optimizer_D)
        self.optimizers.append(self.optimizer_R)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
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
        self.fake_B = self.netT(self.real_A)
        wraped_images, reg_term = self.netR(self.real_A, self.real_B, apply_on=[self.real_A, self.fake_B])
        self.stn_reg_term = reg_term
        self.registered_real_A = wraped_images[0]
        # Registration first -- Then --> Translation
        self.fake_TR_B = self.netT(self.registered_real_A)
        # Translation first  -- Then --> Registration
        self.fake_RT_B = wraped_images[1]
        if self.tb_visualizer:
            with torch.no_grad():
                self.deformation_field_A_to_B = self.netR.module.get_grid(self.real_A, self.real_B)

    def backward_T_and_R(self):
        """Calculate GAN and L1 loss for the translation and registration networks."""
        # Registration first (TR):
        # ----> Reconstruction loss:
        self.loss_L1_TR = self.opt.lambda_recon * self.criterionL1(self.fake_TR_B, self.real_B)
        # ----> GAN loss:
        fake_AB_t = torch.cat((self.real_A, self.fake_TR_B), 1)
        pred_fake = self.netD(fake_AB_t)
        self.loss_GAN_TR = self.opt.lambda_GAN * self.criterionGAN(pred_fake, True)
        # --------> Multi-scale discrimnaotr
        for i in range(self.opt.multi_resolution - 1):
            sh, sw = self.real_A.size(2) // (2 ** (i + 1)), self.real_A.size(3) // (2 ** (i + 1)),
            real_A_resized = F.interpolate(self.real_A, (sh, sw), mode='bilinear', align_corners=False)
            fake_B_B_resized = F.interpolate(self.fake_TR_B, (sh, sw), mode='bilinear', align_corners=False)
            fake_AB_t = torch.cat((real_A_resized, fake_B_B_resized), 1)
            pred_fake = self.netD_multiresolution[i](fake_AB_t)
            self.loss_GAN_TR += self.opt.lambda_GAN * self.criterionGAN(pred_fake, True)

        # Translation First:
        # ----> Reconstruction loss:
        self.loss_L1_RT = self.opt.lambda_recon * self.criterionL1(self.fake_RT_B, self.real_B)

        # ----> GAN loss:
        fake_AB_t = torch.cat((self.real_A, self.fake_RT_B), 1)
        pred_fake = self.netD(fake_AB_t)
        self.loss_GAN_RT = self.opt.lambda_GAN * self.criterionGAN(pred_fake, True)
        # --------> Multi-scale discrimnaotr
        for i in range(self.opt.multi_resolution - 1):
            sh, sw = self.real_A.size(2) // (2 ** (i + 1)), self.real_A.size(3) // (2 ** (i + 1)),
            real_A_resized = F.interpolate(self.real_A, (sh, sw), mode='bilinear', align_corners=False)
            fake_B_P_resized = F.interpolate(self.fake_RT_B, (sh, sw), mode='bilinear', align_corners=False)
            fake_AB_t = torch.cat((real_A_resized, fake_B_P_resized), 1)
            pred_fake = self.netD_multiresolution[i](fake_AB_t)
            self.loss_GAN_RT += self.opt.lambda_GAN * self.criterionGAN(pred_fake, True)

        self.loss_smoothness = self.opt.lambda_smooth * self.stn_reg_term

        loss = self.loss_L1_TR + self.loss_L1_RT + self.loss_GAN_TR + self.loss_GAN_RT + self.loss_smoothness
        loss.backward()

        return loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # --------> Multi-scale discrimnaotr
        for i in range(self.opt.multi_resolution - 1):
            sh, sw = self.real_A.size(2) // (2 ** (i + 1)), self.real_A.size(3) // (2 ** (i + 1)),
            real_A_resized = F.interpolate(self.real_A, (sh, sw), mode='bilinear', align_corners=False)
            real_B_resized = F.interpolate(self.real_B, (sh, sw), mode='bilinear', align_corners=False)
            real_AB = torch.cat((real_A_resized, real_B_resized), 1)
            pred_real = self.netD_multiresolution[i](real_AB)
            loss_D_real += self.criterionGAN(pred_real, True)

        # Registration Firsts (TR):
        # ----> Fake
        fake_AB = torch.cat((self.real_A, self.fake_TR_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake_TR = self.criterionGAN(pred_fake, False)
        # --------> Multi-scale discrimnaotr
        for i in range(self.opt.multi_resolution - 1):
            sh, sw = self.real_A.size(2) // (2 ** (i + 1)), self.real_A.size(3) // (2 ** (i + 1)),
            real_A_resized = F.interpolate(self.real_A, (sh, sw), mode='bilinear', align_corners=False)
            fake_B_B_resized = F.interpolate(self.fake_TR_B, (sh, sw), mode='bilinear', align_corners=False)
            fake_AB_t = torch.cat((real_A_resized, fake_B_B_resized), 1)
            pred_fake = self.netD_multiresolution[i](fake_AB_t.detach())
            self.loss_D_fake_TR += self.criterionGAN(pred_fake, False)

        # Translation First (RT):
        # ----> Fake
        fake_AB = torch.cat((self.real_A, self.fake_RT_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake_RT = self.criterionGAN(pred_fake, False)
        # --------> Multi-scale discrimnaotr
        for i in range(self.opt.multi_resolution - 1):
            sh, sw = self.real_A.size(2) // (2 ** (i + 1)), self.real_A.size(3) // (2 ** (i + 1)),
            real_A_resized = F.interpolate(self.real_A, (sh, sw), mode='bilinear', align_corners=False)
            fake_B_P_resized = F.interpolate(self.fake_RT_B, (sh, sw), mode='bilinear', align_corners=False)
            fake_AB_t = torch.cat((real_A_resized, fake_B_P_resized), 1)
            pred_fake = self.netD_multiresolution[i](fake_AB_t.detach())
            self.loss_D_fake_RT += self.criterionGAN(pred_fake, False)

        # combine loss and calculate gradients
        self.loss_D = 0.5 * self.opt.lambda_GAN * (loss_D_real + self.loss_D_fake_TR + self.loss_D_fake_RT)
        self.loss_D.backward()

        return self.loss_D

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # TR(I_a) and RT(I_a)
        # Backward D
        self.set_requires_grad([self.netT, self.netR], False)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.set_requires_grad([self.netT, self.netR], True)

        # Backward translation and registration networks
        self.set_requires_grad([self.netD, *self.netD_multiresolution], False)
        self.optimizer_R.zero_grad()
        self.optimizer_T.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_T_and_R()  # calculate gradients for translation and registration networks
        self.optimizer_R.step()
        self.optimizer_T.step()
        self.set_requires_grad([self.netD, *self.netD_multiresolution], True)

        # Update tb visualizer on each iteration step - if enabled
        if self.tb_visualizer is not None:
            self.tb_visualizer.iteration_step()
