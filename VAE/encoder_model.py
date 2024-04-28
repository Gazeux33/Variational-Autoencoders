import torch
from torch import nn


class EncoderModelV0(nn.Module):
    def __init__(self, z_dim):
        super(EncoderModelV0, self).__init__()
        self.z_dim = z_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )
        self.linear = nn.Linear(256 * 2 * 2, self.z_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear(x)
        return x


class EncoderModelV1(nn.Module):
    def __init__(self, z_dim):
        super(EncoderModelV1, self).__init__()
        self.z_dim = z_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )
        self.z_mean_layer = nn.Linear(256 * 2 * 2, self.z_dim)
        self.z_log_var_layer = nn.Linear(256 * 2 * 2, self.z_dim)
        self.sampling_layer = SamplingLayer()

    def forward(self, x):
        x = self.conv_layers(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling_layer(z_mean, z_log_var)
        return z_mean, z_log_var, z


class SamplingLayer(nn.Module):
    def __init__(self):
        super(SamplingLayer, self).__init__()

    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(size=(batch_size, dim))
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class EncoderModelV2(nn.Module):
    def __init__(self, z_dim):
        super(EncoderModelV2, self).__init__()
        self.z_dim = z_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )
        self.z_mean_layer = nn.Linear(256 * 2 * 2, self.z_dim)
        self.z_log_var_layer = nn.Linear(256 * 2 * 2, self.z_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var
