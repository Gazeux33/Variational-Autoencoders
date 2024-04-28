from torch import nn
import torch
import torch.nn.functional as F


class AutoEncoderModelV0(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoderModelV0, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z


class VariationalAutoEncoderModelV0(nn.Module):
    def __init__(self, encoder, decoder, kl_weight):
        super(VariationalAutoEncoderModelV0, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean



