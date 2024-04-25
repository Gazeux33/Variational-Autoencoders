from torch import nn


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
    def __init__(self, encoder, decoder):
        super(VariationalAutoEncoderModelV0, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

