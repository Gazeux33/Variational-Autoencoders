from torch import nn


class DecoderModelV0(nn.Module):
    def __init__(self, z_dim):
        super(DecoderModelV0, self).__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, 256 * 2 * 2),
            nn.Unflatten(1, (256, 2, 2))
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.conv_layers(x)
        return x
