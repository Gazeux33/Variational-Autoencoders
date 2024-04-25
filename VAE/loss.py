from torch import nn
import torch
import torch.nn.functional as F


class VaeLossV0(nn.Module):
    def __init__(self, loss_fn, beta):
        super(VaeLossV0, self).__init__()
        self.loss_fn = loss_fn
        self.beta = beta

    def forward(self, data, reconstruction, z_mean, z_log_var):
        recon_loss = self.reconstruction_loss(data, reconstruction)
        kl_loss_val = self.kl_loss(z_mean, z_log_var)
        total_loss_val = recon_loss + kl_loss_val
        return total_loss_val

    def reconstruction_loss(self, data, reconstruction):
        bce_loss = F.binary_cross_entropy(reconstruction, data, reduction='none')
        bce_loss = torch.mean(torch.sum(bce_loss, dim=(1, 2, 3)))
        bce_loss *= self.beta
        return bce_loss

    @staticmethod
    def kl_loss(z_mean, z_log_var):
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss
