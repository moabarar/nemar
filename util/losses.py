import torch


def smoothness_loss(deformation, img=None, alpha=0.0):
    def tmp(x):
        return torch.abs(x)

    diff_1 = tmp(deformation[:, :, 1::, :] - deformation[:, :, 0:-1, :])
    diff_2 = tmp((deformation[:, :, :, 1::] - deformation[:, :, :, 0:-1]))
    diff_3 = tmp(deformation[:, :, 0:-1, 0:-1] - deformation[:, :, 1::, 1::])
    diff_4 = tmp(deformation[:, :, 0:-1, 1::] - deformation[:, :, 1::, 0:-1])
    if img is not None and alpha > 0.0:
        def f(x):
            return torch.exp(x)

        mask = img
        weight_1 = f(-alpha * torch.abs(mask[:, :, 1::, :] - mask[:, :, 0:-1, :]))
        weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_2 = f(- alpha * torch.abs(mask[:, :, :, 1::] - mask[:, :, :, 0:-1]))
        weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_3 = f(- alpha * torch.abs(mask[:, :, 0:-1, 0:-1] - mask[:, :, 1::, 1::]))
        weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_4 = f(- alpha * torch.abs(mask[:, :, 0:-1, 1::] - mask[:, :, 1::, 0:-1]))
        weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    else:
        weight_1 = weight_2 = weight_3 = weight_4 = 1.0

    def my_mean(x):
        return torch.mean(x)

    loss = my_mean(weight_1 * diff_1) + my_mean(weight_2 * diff_2) + my_mean(weight_3 * diff_3) + my_mean(
        weight_4 * diff_4)
    return loss
