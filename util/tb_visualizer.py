import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardVisualizer:
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--tbvis_iteration_update_rate', type=int, default=1000,
                                help='Number of iterations steps before writing statistics to tensorboard.')
            parser.add_argument('--tbvis_disable_report_weights', action='store_true',
                                help='Whether to not report the network weights change')
            parser.add_argument('--tbvis_disable_report_offsets', action='store_true',
                                help='Whether to not report mean deformation offsets in x and y direction.')
        return parser

    def __init__(self, mirnet_model, networks_names, losses_names, opt):
        self.writer = None
        self.writer_log_dir = '{}/{}/{}_tensorboard_logs'.format(opt.checkpoints_dir, opt.name, opt.name)
        self.enabled = False
        self.model = mirnet_model
        self.networks_names = networks_names
        self.networks = {}
        self.iteration_update_rate = opt.tbvis_iteration_update_rate
        self.iteration_cnt = 0
        self.save_count = 0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_cnt = 1
        self.losses_names = losses_names
        self.report_weights = not opt.tbvis_disable_report_weights
        self.report_offsets = not opt.tbvis_disable_report_offsets

    def save_current_weights(self):
        for net_name, net in self.networks.items():
            for n, p in net.named_parameters():
                if p.requires_grad:
                    suffix = 'Bias' if ("bias" in n) else 'Weight'
                    name = '{}/data/{}/{}'.format(net_name, suffix, n)
                    self.writer.add_histogram(name, p.clone().cpu().data.numpy(), self.save_count)

    def save_histogram(self, name, tensor):
        if name not in self.image_step:
            self.image_step[name] = 0
            self.image_count[name] = -1
        self.image_count[name] = (self.image_count[name] + 1) % self.grads_update_rate
        if self.image_count[name] != 0:
            return
        step = self.image_step[name] + 1
        self.image_step[name] = step
        tensor = tensor.detach().cpu().numpy()
        self.writer.add_histogram(name, tensor, step)

    def save_current_losses(self):
        for lname in self.losses_names:
            loss_val = getattr(self.model, 'loss_{}'.format(lname))
            self.writer.add_scalar('loss/{}'.format(lname), loss_val, self.save_count)

    def save_offsets(self):
        mean_x = self.offset_x / self.offset_cnt
        self.writer.add_scalar('offset/mean_x', mean_x, self.save_count)
        mean_y = self.offset_y / self.offset_cnt
        self.writer.add_scalar('offset/mean_y', mean_y, self.save_count)
        self.offset_x = self.offset_y = 0.0
        self.offset_cnt = 0.0

    def iteration_step(self):
        if not self.enabled:
            return
        if self.report_offsets:
            offset = self.model.deformation_field_A_to_B.data.cpu().numpy()
            self.offset_x += np.mean(offset[:, 0, ...])
            self.offset_y += np.mean(offset[:, 1, ...])
            self.offset_cnt += 1
        if self.iteration_update_rate <= 0:
            return
        if self.iteration_cnt == 0:
            self.save_current_losses()
            if self.report_weights:
                self.save_current_weights()
            if self.report_offsets:
                self.save_offsets()
            self.save_count += 1
        self.iteration_cnt = (self.iteration_cnt + 1) % self.iteration_update_rate

    def epoch_step(self):
        # Don't report statistics if the update is in iteration resolution.
        if not self.enabled or self.iteration_update_rate > 0:
            return
        self.save_current_losses()
        if self.report_weights:
            self.save_current_weights()
        if self.report_offsets:
            self.save_offsets()
        self.save_count += 1

    def end(self):
        self.writer.close()

    def enable(self):
        self.enabled = True
        self.writer = SummaryWriter(self.writer_log_dir)
        for net_name in self.networks_names:
            self.networks[net_name] = getattr(self.model, net_name)
