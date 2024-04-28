from torch import nn
import torch
import torch.nn.functional as F


class VaeLossV0(nn.Module):
    def __init__(self, beta):
        super(VaeLossV0, self).__init__()
        self.beta = beta

    def forward(self, data, reconstruction, z_mean, z_log_var):
        recon_loss = self.reconstruction_loss(data, reconstruction)
        kl_loss_val = self.kl_loss(z_mean, z_log_var)
        total_loss_val = recon_loss + kl_loss_val
        return total_loss_val,recon_loss,kl_loss_val

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


class VaeLossV1(nn.Module):
    def __init__(self, loss_fn, beta, f_kl=1, f_rec=1):
        super(VaeLossV1, self).__init__()
        self.beta = beta
        self.loss_fn = loss_fn
        self.f_kl = f_kl
        self.f_rect = f_rec

    def forward(self, data, reconstruction, z_mean, z_log_var):
        recon_loss = self.reconstruction_loss(data, reconstruction)
        kl_loss_val = self.kl_loss(z_mean, z_log_var)
        total_loss_val = recon_loss + kl_loss_val
        return total_loss_val,recon_loss,kl_loss_val

    def reconstruction_loss(self, data, reconstruction):
        bce_loss = self.loss_fn(reconstruction, data)
        return bce_loss * self.f_rect

    def kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss * self.f_kl


class VaeLossV2(nn.Module):
    def __init__(self,kld_weight=10.0):
        super(VaeLossV2, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, data, reconstruction, mean, log_var):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, data)

        # kld loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim=1), dim=0)

        # Total loss
        total_loss_val = recon_loss + kld_loss * self.kld_weight
        return total_loss_val, recon_loss, kld_loss
