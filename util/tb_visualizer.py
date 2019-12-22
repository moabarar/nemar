import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from torch.utils.tensorboard import SummaryWriter


def plot_grad_flow(named_parameters, save_name, limit=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            tmp = str(n).split('.')
            name = '{}_W'.format(tmp[-2])
            layers.append(name)
            ave_grads.append(p.grad.mean())
            max_grads.append(0.0)  # p.grad.max())
            min_grads.append(0.0)  # p.grad.min())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c")
    plt.bar(np.arange(len(min_grads)), min_grads, alpha=0.4, lw=1, color="m")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.2, lw=1, color="y")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=8)
    plt.xlim(left=0, right=len(ave_grads))
    if limit is not None:
        plt.ylim(bottom=limit[0], top=limit[1])  # zoom in on the lower gradient regions
    plt.xlabel("Layers", fontsize=8)
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="m", lw=4),
                Line2D([0], [0], color="y", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'min-gradient', 'mean-gradient', 'zero-gradient', ])
    plt.tight_layout()
    plt.savefig(save_name)
    plt.clf()


def write_grads(writer, named_parameters, prefix, niter):
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            tmp = str(n).split('.')
            name = '{}/{}_W'.format(prefix, tmp[-2])
            writer.add_histogram(name, p.clone().cpu().grad.numpy(), niter)


def get_parameter_name(name):
    # tmp = name.split('.')
    # return 'block{}_{}'.format(len(tmp) - 3, tmp[-2])
    return name


class TensorboardVisualizer:
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--tbvis_iteration_update_rate', type=int, default=-1,
                                help='Number of iterations steps before updating tensorboard visualization step.')
            parser.add_argument('--tbvis_grads_update_rate', type=int, default=100, help='Number of iterations for '
                                                                                         'grads visualization.')

        return parser

    def __init__(self, model, networks_names, losses_names, opt):
        self.writer = None
        self.writer_log_dir = './runs/data-{}_cfg-{}_lr-{}_stnlr-{}_identity-{}_edge-{}_l1-{}'.format(
            opt.dataroot.split('/')[-2],
            opt.stn_cfg,
            opt.lr,
            opt.stn_lr,
            opt.lambda_stn_reg,
            opt.lambda_edge_loss,
            opt.lambda_L1)
        self.enabled = False
        self.model = model
        self.networks_names = networks_names
        self.networks = {}
        self.iteration_update_rate = opt.tbvis_iteration_update_rate
        self.iteration_cnt = 0
        self.save_count = 0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_cnt = 1
        self.grads_update_rate = opt.tbvis_grads_update_rate
        self.grads_cnt = 0
        self.grads_save_count = 0
        self.image_count = {}
        self.image_step = {}
        self.losses_names = losses_names

    def save_current_grads(self):
        if self.grads_update_rate < 0:
            return
        if self.grads_cnt == 0:
            for net_name, net in self.networks.items():
                for n, p in net.named_parameters():
                    if p.requires_grad and ("bias" not in n):
                        tmp = get_parameter_name(n)
                        name = '{}/grads/{}_W'.format(net_name, tmp)
                        x = p.grad.clone().cpu().numpy()
                        # if net_name is 'netSTN_A' : print('{} : {}'.format(name, x))
                        self.writer.add_histogram(name, x, self.grads_save_count)
            self.grads_save_count += 1
        self.grads_cnt = (self.grads_cnt + 1) % self.grads_update_rate

    def save_current_weights(self):
        for net_name, net in self.networks.items():
            for n, p in net.named_parameters():
                if p.requires_grad:
                    suffix = 'Bias' if ("bias" in n) else 'Weight'
                    tmp = get_parameter_name(n)
                    name = '{}/data/{}/{}'.format(net_name, suffix, tmp)
                    self.writer.add_histogram(name, p.clone().cpu().data.numpy(), self.save_count)

    def save_image(self, name, tensor):
        if name not in self.image_step:
            self.image_step[name] = 0
            self.image_count[name] = -1
        self.image_count[name] = (self.image_count[name] + 1) % self.grads_update_rate
        if self.image_count[name] != 0:
            return
        step = self.image_step[name] + 1
        self.image_step[name] = step

        def normalize(x):
            if np.min(x) == np.max(x):
                return np.zeros_like(x)
            else:
                return x - np.min(x) / (np.max(x) - np.min(x))

        tensor = tensor.detach().cpu().numpy()
        tensor = normalize(np.linalg.norm(tensor[0, ...], axis=0, keepdims=True))
        self.writer.add_image(name, tensor, step)

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
        print('mean_x: {} ,  mean_y: {}'.format(mean_x, mean_y))

    def iteration_step(self):
        if not self.enabled:
            return
        offset = self.model.offset_A.data.cpu().numpy()
        self.offset_x += np.mean(offset[:, 0, ...])
        self.offset_y += np.mean(offset[:, 1, ...])
        self.offset_cnt += 1
        if self.iteration_update_rate <= 0:
            return
        if self.iteration_cnt == 0:
            self.save_current_weights()
            self.save_current_losses()
            self.save_offsets()
            self.save_count += 1

            self.offset_x = 0.0
            self.offset_y = 0.0
        self.iteration_cnt = (self.iteration_cnt + 1) % self.iteration_update_rate

    def epoch_step(self):
        if not self.enabled or self.iteration_update_rate > 0:
            return
        self.save_current_weights()
        self.save_current_losses()
        self.save_offsets()
        self.save_count += 1

    def end(self):
        self.writer.close()

    def enable(self):
        self.enabled = True
        self.writer = SummaryWriter(self.writer_log_dir)
        for net_name in self.networks_names:
            self.networks[net_name] = getattr(self.model, net_name)
