import torch.nn as nn
import torch.nn.functional as F
import torch

from config import selected_config as conf

#
# The VAE is comprised of an encoder and a decoder both with
# two feedforward hidden layers of 128 units. The hyperbolic tangent
# activation function is applied to all hidden layers, except the output
# layer. The dimension of the latent space L is fixed at 16. The noise-
# aware encoder has the same structure as the speech-based encoder of the
# standard VAE. The fully supervised DNN-WF contains 5 hidden layers,
# each with 128 units, and its architecture is built to contain a similar
# number of parameters as our VAE model. No temporal information is
# considered in DNN-WF, which is consistent with the non-sequential
# characteristic of the VAE. We apply the ReLU activation function to
# all hidden layers, and the sigmoid function is put on the output layer
# to ensure the estimate of the Wiener filter mask lies in the range [0, 1].
# The parameters θ and φ of the VAE are optimized by Adam [25] with a
# learning rate of 1e-3, and the parameters γ of the noise-aware encoder
# with a learning rate of 1e-4.


# latent space sdi 16
# 2 hidden layers of 128 units per l'encoder e per il decoder
# full connected

class SimpleEncoder(nn.Module):

    def __init__(self, F=513, H=128, L=16):

        super().__init__()

        self.fc1 = nn.Linear(in_features=F, out_features=H)
        self.fc_mean = nn.Linear(in_features=H, out_features=L)
        self.fc_log_var = nn.Linear(in_features=H, out_features=L)

    def forward(self, x):

        x = F.tanh(self.fc1(x))

        mean = F.tanh(self.fc_mean(x))
        log_var = F.tanh(self.fc_log_var(x))

        # Create sample from standard normal distribution
        sample = torch.randn_like(mean)
        sample.to(x.device)
        # Reparameterization trick (mean + std * N(0,1),
        # multiply log_var by 0.5 because std = exp(0.5 * log_var) = sqrt(var)
        z = mean + torch.exp(0.5 * log_var) * sample
        return z, mean, log_var

class SimpleDecoder(nn.Module):

    def __init__(self, F=513, H=128, L=16):

        super().__init__()

        self.fc2 = nn.Linear(in_features=L, out_features=H)
        self.fc3 = nn.Linear(in_features=H, out_features=F)
    def forward(self, z):
        x = F.tanh(self.fc2(z))
        x = self.fc3(x)
        return x


class SimpleVAE(nn.Module):
    def __init__(self, F=513, H=128, L=16):
        super().__init__()
        self.encoder = SimpleEncoder(F, H, L)
        self.decoder = SimpleDecoder(F, H, L)

    def forward(self, x):
        x, mean, log_var = self.encoder(x)
        x_hat = self.decoder(x)
        return x_hat, mean, log_var

if __name__ == "__main__":
    vae  = SimpleVAE()
    vae.to("cuda")
    rand_sample = torch.randn(3, 513)
    rand_sample = rand_sample.to("cuda")
    x_hat, mean, log_var = vae(rand_sample)
    print(x_hat.shape)