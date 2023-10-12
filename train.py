from loader import load_dataset, AudioDataset
from torch.utils.data import DataLoader
from config import selected_config as conf
from model import VariationalAutoEncoder
import numpy as np
import torch
from base_trainer import BaseTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
conf.INTERVAL_TENSORBOARD = 100

if torch.cuda.is_available():
    print("Using CUDA :)")
else:
    print("Using CPU :(")

class Trainer(BaseTrainer):
    def __init__(self, *args, alpha=0.5, beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha # Weight of the KL loss
        self.beta = beta # Weight of the angle loss
        self.init_dataloader()
        self.init_models()

    def init_dataloader(self):


        # Load dataset
        spectrograms = load_dataset('dataset/spectrograms')

        # Shuffle the list
        np.random.shuffle(spectrograms)
        train = spectrograms[:int(len(spectrograms) * 0.8)]
        test = spectrograms[int(len(spectrograms) * 0.8):]
        train_dataset = AudioDataset(train, length=conf.SIGNAL_LENGTH)
        test_dataset = AudioDataset(test, length=conf.SIGNAL_LENGTH)
        self.train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        self.test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    def init_models(self):

        # Load model
        self.autoencoder = VariationalAutoEncoder()
        self.autoencoder.to(self.device)
        self.models = [self.autoencoder]

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        self.optimizers = [self.optimizer]
        self.loss = torch.nn.MSELoss()

    def update_lr(self):
        self.lr = 1e-3



    def train_epoch(self):
        hist_loss = []
        hist_rec_loss = []
        hist_kl_loss = []
        for i, (clean_Tmag, clean_Tphase, noised_Tmag, noised_Tphase) in enumerate(self.train_dataloader):
            # Update counter
            self.idx += 1

            # Move to device
            clean_Tmag = clean_Tmag.to(self.device)
            clean_Tphase = clean_Tphase.to(self.device)
            noised_Tphase = noised_Tphase.to(self.device)
            noised_Tmag = noised_Tmag.to(self.device)

            # Apply model
            Tmag_hat, Tphase_hat, mean, log_var = self.autoencoder(noised_Tmag, noised_Tphase)

            # Calculate loss

            # We need a special loss for angles (ex: difference between epsilon and 2 pi - epsilon has to be small)
            phase_diff = Tphase_hat - clean_Tphase
            # phase_diff2 = clean_Tphase - Tphase_hat
            # phase_diff = torch.where(torch.abs(phase_diff) < np.pi, phase_diff, phase_diff2)
            # angle_loss = torch.var(phase_diff)
            # Or using the chord
            chord_loss = torch.mean((1 - torch.cos(phase_diff))**2 + torch.sin(phase_diff)**2)
            reconstruction_loss = (1 - self.beta) * self.loss(Tmag_hat, clean_Tmag) + self.beta * chord_loss

            # kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            kl_loss = (torch.exp(log_var)**2 + mean**2 - log_var - 0.5).mean()
            loss = (1 - self.alpha) * reconstruction_loss + self.alpha * kl_loss
            hist_loss.append(loss.item())
            hist_rec_loss.append(reconstruction_loss.item())
            hist_kl_loss.append(kl_loss.item())
            if i % conf.INTERVAL_TENSORBOARD == 0:
                self.mean_train_loss = np.mean(hist_loss)
                print("----------------------------------------------")
                print(f"Loss {i} iteration: {self.mean_train_loss:.2f}")
                print(f"Reconstruction loss {i} iteration: {np.mean(hist_rec_loss):.2f}")
                print(f"KL loss {i} iteration: {np.mean(hist_kl_loss):.2f}")
                hist_loss = []
                hist_rec_loss = []
                hist_kl_loss = []
                self.log_tensorboard()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    def train(self):

        autoencoder = VariationalAutoEncoder()
        for epoch in range(100):
            print(f"Epoch {epoch}")
            self.train_epoch()





if __name__ == "__main__":

    trainer = Trainer()
    trainer.train()