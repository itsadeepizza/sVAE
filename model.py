import torch.nn as nn
import torch.nn.functional as F
import torch

from config import selected_config as conf


# INPUT: 2 x SIGNAL_LENGTH x FREQ_LENGTH
#
# CONV1: 16 x 50 x 50
#
# CONV2: 32 x 25 x 25
#
# PRE-LATENT: 2 x 64 (first channel for mean, second for log variance)
# -- sampling --
# LATENT: 64
#
# CONV2: 32 x 25 x 25
#
# CONV1: 16 x 50 x 50
#
# OUTPUT: 2 x SIGNAL_LENGTH x FREQ_LENGTH

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=10, stride=5)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)

        self.fc_mean = nn.Linear(in_features=32 * 23 * 23, out_features=64)

        self.fc_log_var = nn.Linear(in_features=32 * 23 * 23, out_features=64)


    def forward(self, Tmag, Tphase):
        Tmag = Tmag.unsqueeze(1)
        Tphase = Tphase.unsqueeze(1)
        x = torch.cat((Tmag, Tphase), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 23 * 23)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)

        # Create sample from standard normal distribution
        sample = torch.randn_like(mean)
        # Reparameterization trick (mean + std * N(0,1),
        # multiply log_var by 0.5 because std = exp(0.5 * log_var) = sqrt(var)
        z = mean + torch.exp(0.5 * log_var) * sample
        return z, mean, log_var



class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=64, out_features=32 * 23 * 24)
        self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=2,  kernel_size=[10, 10], padding=[0,1], stride=5)

    def forward(self, latent):
        x = F.relu(self.fc(latent))
        x = x.view(-1, 32, 23, 24)
        x = F.relu(self.conv1(x))
        # Add a padding of 1 on the top for the second dimension only

        x = self.conv2(x)
        # Crop the output to the desired size
        x = x[:, :, :, :257]
        Tmag = x[:, 0, :, :]
        Tphase = x[:, 1, :, :]
        return Tmag, Tphase



class VariationalAutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, Tmag, Tphase):
        latent, mean, log_var = self.encoder(Tmag, Tphase)
        Tmag_hat, Tphase_hat = self.decoder(latent)
        return Tmag_hat, Tphase_hat, mean, log_var
