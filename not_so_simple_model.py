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
# standard VAE.
#
# The fully supervised DNN-WF contains 5 hidden layers,
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
# NC: New Changes from simple_model:
# NC: leakyReLu activation
# NC: added batchnorm
# NC: 1 additional layer of 64 
# NC: dropout (after activation)
# NC: function to sample normal gaussian
# NC: function to sample from learned (mu,var) pairs
# Two (discussed) ways to generate new sample: 
# a) from the normal distribution (as we assume the prior distribution is gaussian)
# ... but this may produce not-good samples (depending on the importance given to the KL)
# b) from the learned (mu,var) 
# ... this implies sampling something similar from a given input
# as the (mu,var) are different for each input
# and we chose to sample from that


class NSSimpleEncoder(nn.Module):

    def __init__(self, F=513, H=254, H2=128, L=64):
        super().__init__()

        self.fc1 = nn.Linear(in_features=F, out_features=H)
        self.bn1 = nn.BatchNorm1d(H)
        self.fc2 = nn.Linear(in_features=H, out_features=H2)
        self.bn2 = nn.BatchNorm1d(H2)
        self.fc_mean = nn.Linear(in_features=H2, out_features=L)
        self.fc_log_var = nn.Linear(in_features=H2, out_features=L)
        self.activation = nn.LeakyReLU(0.1)
        # Add dropout
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)

        # Create sample from standard normal distribution
        sample = torch.randn_like(mean)
        sample.to(x.device)
        # Reparameterization trick (mean + std * N(0,1),
        # multiply log_var by 0.5 because std = exp(0.5 * log_var) = sqrt(var)
        z = mean + torch.exp(0.5 * log_var) * sample
        # z = mean
        return z, mean, log_var


class NSSimpleDecoder(nn.Module):

    def __init__(self, F=513, H=254, H2=128, L=64):
        super().__init__()

        self.fc2 = nn.Linear(in_features=L, out_features=H2)
        self.bn1 = nn.BatchNorm1d(H2)
        self.fc3 = nn.Linear(in_features=H2, out_features=H)
        self.bn2 = nn.BatchNorm1d(H)
        self.fc4 = nn.Linear(in_features=H, out_features=F)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, z):
        x = self.activation(self.fc2(z))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class NSSimpleVAE(nn.Module):
    def __init__(self, F=513, H=128, H2=64, L=16):
        super().__init__()
        self.encoder = NSSimpleEncoder(F, H, H2, L)
        self.decoder = NSSimpleDecoder(F, H, H2, L)

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

    def normal_sampling(self, sample_size=1, L=16):
        # Samples sample_size samples from normal distribution with dimension L
        z = torch.randn(sample_size, L)
        return (z)

    def learned_sampling(self, similar_example, sample_size=(1, 1)):
        # Samples each element according to the learned pair (mean, var)
        # in the hidden space     
        _, mu, log_var = self.encoder(similar_example)
        z = torch.normal(mean=mu, std=torch.exp(0.5 * log_var), size=sample_size)
        return z


if __name__ == "__main__":
    vae = NSSimpleVAE()
    vae.to("cuda")
    rand_sample = torch.randn(3, 513)
    rand_sample = rand_sample.to("cuda")
    x_hat, mean, log_var = vae(rand_sample)
    print(x_hat.shape)
