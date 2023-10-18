from loader import load_dataset, AudioDataset
from torch.utils.data import DataLoader
from config import selected_config as conf
from model import VariationalAutoEncoder
import numpy as np
import torch
from base_trainer import BaseTrainer
import os





class Trainer(BaseTrainer):
    def __init__(self, *args, alpha=0.5, beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha # Weight of the KL loss
        self.beta = beta # Weight of the angle loss
        self.init_dataloader()
        self.init_models()

        self.rec_loss = 0
        self.kl_loss = 0
        self.mag_loss = 0
        self.phase_loss = 0

    def init_dataloader(self):


        # Load dataset
        spectrograms = load_dataset('dataset/spectrograms')

        # Shuffle the list
        np.random.shuffle(spectrograms)
        train = spectrograms[:int(len(spectrograms) * 0.8)]
        test = spectrograms[int(len(spectrograms) * 0.8):]
        train_dataset = AudioDataset(train, length=conf.SIGNAL_LENGTH)
        test_dataset = AudioDataset(test, length=conf.SIGNAL_LENGTH)
        self.train_dataloader = DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=0)
        self.test_dataloader = DataLoader(test_dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=0)

    def init_models(self):

        # Load model
        self.autoencoder = VariationalAutoEncoder()
        self.autoencoder.to(self.device)
        self.models = [self.autoencoder]

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        if conf.LOAD_PATH is not None:
            # I added "map_location=conf.DEVICE" to deal with error in google colab when gpu is
            # not available

            vae_w = torch.load(os.path.join(conf.LOAD_PATH, f"VariationalAutoEncoder/VariationalAutoEncoder_{conf.LOAD_IDX}.pth"), map_location=conf.DEVICE)
            self.autoencoder.load_state_dict(vae_w)
            # Update logs


    def update_lr(self):
        self.lr = 1e-3



    def train_epoch(self):
        for i, (clean_Tmag, clean_Tphase, noised_Tmag, noised_Tphase) in enumerate(self.train_dataloader):
            # Update counter
            self.idx += conf.BATCH_SIZE

            # Train a single sample
            T_mag_hat, t_phase_hat, *_ = self.train_sample(clean_Tmag, clean_Tphase, noised_Tmag, noised_Tphase)


            if self.idx % conf.INTERVAL_SAVE_MODEL < conf.BATCH_SIZE:
                # Save models as pth
                self.save_models()

            if self.idx % conf.INTERVAL_TENSORBOARD < conf.BATCH_SIZE:
                # self.mean_train_loss = np.mean(hist_loss)
                print("----------------------------------------------")
                print(f"Index {self.idx}")
                print(f"Mean train loss {self.mean_train_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean rec loss {self.rec_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean kl loss {self.kl_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean mag loss {self.mag_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean phase loss {self.phase_loss / conf.INTERVAL_TENSORBOARD}")
                print("----------------------------------------------")
                self.log_tensorboard()

            if self.idx % conf.INTERVAL_TENSORBOARD_PLOT < conf.BATCH_SIZE:
                # Show reconstucted spectrogramon tensorboard
                import matplotlib.pyplot as plt
                # Convert Tmag to image
                fig, ax = plt.subplots(2, 2)
                ax[0][0].imshow(T_mag_hat[0].detach().to("cpu").numpy().T)
                ax[0][1].imshow(clean_Tmag[0].detach().to("cpu").numpy().T)
                ax[1][0].imshow(t_phase_hat[0].detach().to("cpu").numpy().T)
                ax[1][1].imshow(clean_Tphase[0].detach().to("cpu").numpy().T)
                # Add labels
                ax[0][0].set_title("Reconstructed magnitude")
                ax[0][1].set_title("Clean magnitude")
                ax[1][0].set_title("Reconstructed phase")
                ax[1][1].set_title("Clean phase")
                img = self.plot_to_tensorboard(fig)
                self.writer.add_image("reconstructed", img, self.idx)


    def log_tensorboard(self):
        """Log data to tensorboard"""

        super().log_tensorboard()
        self.writer.add_scalar("rec_loss", self.rec_loss / conf.INTERVAL_TENSORBOARD, self.idx)
        self.writer.add_scalar("kl_loss", self.kl_loss / conf.INTERVAL_TENSORBOARD, self.idx)
        self.writer.add_scalar("mag_loss", self.mag_loss / conf.INTERVAL_TENSORBOARD, self.idx)
        self.writer.add_scalar("phase_loss", self.phase_loss / conf.INTERVAL_TENSORBOARD, self.idx)
        self.rec_loss = 0
        self.kl_loss = 0
        self.mag_loss = 0
        self.phase_loss = 0

    def train_sample(self, clean_Tmag, clean_Tphase, noised_Tmag, noised_Tphase):
        """Train a single sample"""
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
        phase_loss = torch.mean((1 - torch.cos(phase_diff)) ** 2 + torch.sin(phase_diff) ** 2)
        mag_loss = self.loss(Tmag_hat, clean_Tmag)
        reconstruction_loss = (1 - self.beta) * mag_loss + self.beta * phase_loss

        # kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_loss = (torch.exp(log_var) ** 2 + mean ** 2 - log_var - 0.5).mean()
        loss = (1 - self.alpha) * reconstruction_loss + self.alpha * kl_loss
        self.mean_train_loss += loss.item()
        self.rec_loss += reconstruction_loss.item()
        self.kl_loss += kl_loss.item()
        self.mag_loss += mag_loss.item()
        self.phase_loss += phase_loss.item()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return  Tmag_hat.detach(), Tphase_hat.detach(), mean.detach(), log_var.detach()


    def train(self):

        autoencoder = VariationalAutoEncoder()
        for epoch in range(10000):
            print(f"Epoch {epoch}")
            self.train_epoch()






if __name__ == "__main__":
    conf.INTERVAL_SAVE_MODEL = 50000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    conf.INTERVAL_TENSORBOARD = 100
    conf.INTERVAL_TENSORBOARD_PLOT = 1000
    conf.BATCH_SIZE = 7
    # conf.LOAD_PATH = r"C:\Users\p.menegatti\PycharmProjects\svae\runs\fit\23-10-18-23H32_purple_place\models"
    # conf.LOAD_IDX = 1000


    trainer = Trainer()
    trainer.train()

# Command to run tensorboard
# tensorboard --logdir=runs/fit