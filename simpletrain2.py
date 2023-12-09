from simple_loader import AudioDataset
from torch.utils.data import DataLoader
from config import selected_config as conf
from simple_model import SimpleVAE
import numpy as np
import torch
from base_trainer import BaseTrainer
import os
import soundfile
import librosa
import scipy.signal



class Trainer(BaseTrainer):
    def __init__(self, *args, alpha:float=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha # Weight of the KL loss
        self.init_dataloader()
        self.init_models()

        self.rec_loss = 0
        self.kl_loss = 0
        self.mag_loss = 0
        self.phase_loss = 0

    def init_dataloader(self):

        dataset = AudioDataset("dataset/unprocessed")
        self.train_dataloader  = DataLoader(dataset, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=5)

    def init_models(self):

        # Load model
        self.autoencoder = SimpleVAE()
        self.autoencoder.to(self.device)
        self.models = [self.autoencoder]

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        if conf.LOAD_PATH is not None:
            # I added "map_location=conf.DEVICE" to deal with error in google colab when gpu is
            # not available

            vae_w = torch.load(os.path.join(conf.LOAD_PATH, f"SimpleVAE/SimpleVAE_{conf.LOAD_IDX}.pth"), map_location=conf.DEVICE)
            self.autoencoder.load_state_dict(vae_w)
            # Update logs


    def update_lr(self):
        # self.lr = 1e-4

        delay_lr = conf.LR_DELAY
        if self.idx < delay_lr:
            self.lr = conf.LR_INIT
        else:
            self.lr = conf.LR_INIT * (conf.LR_DECAY ** ((self.idx - delay_lr) // conf.LR_STEP))
        # or in one func conf.LR_INIT * (conf.LR_DECAY ** (sigmoid(x-100_000)*(self.idx - delay_lr) // conf.LR_STEP))
        # lr is multiplied by LR_DECAY every LR_STEP steps



    def train_epoch(self):
        for i, (clean_sig, noised_sig, phase, torch_label) in enumerate(self.train_dataloader):
            # Update counter
            self.idx += conf.BATCH_SIZE

            # Ivert torch.tensor([ord(c) for c in path_audio], dtype=torch.float32) to get the string again
            label = "".join([chr(int(c)) for c in torch_label[0]])

            # Train a single sample
            # T_mag_hat, t_phase_hat, *_ = self.train_sample(clean_Tmag, clean_Tphase, noised_Tmag, noised_Tphase)
            #  Try to reconstruct the original signal (for DEBUG PURPOSES)
            denoise_sig, *_ = self.train_sample(clean_sig, noised_sig, label)





            if self.idx % conf.INTERVAL_UPDATE_LR < conf.BATCH_SIZE:
                # UPDATE LR
                self.update_lr()
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr


            if self.idx % conf.INTERVAL_SAVE_MODEL < conf.BATCH_SIZE:
                # Save models as pth
                self.save_models()

            if self.idx % conf.INTERVAL_TENSORBOARD < conf.BATCH_SIZE:
                # self.mean_train_loss = np.mean(hist_loss)

                # TRY TO RECONSTRUCT THE SIGNAL
                # Invert STFT
                phase = phase.cpu().numpy()
                phase = phase.reshape(-1, 101)
                denoise_sig = denoise_sig.cpu().numpy()
                denoise_sig = denoise_sig.swapaxes(0, 1)
                clean_sig = clean_sig.cpu().numpy()
                clean_sig = clean_sig.reshape(-1, 101)

                K = 1024 * 25
                win_length = 1024
                window = scipy.signal.windows.cosine(win_length)
                rec_signal_hat = librosa.istft(denoise_sig * phase, length=K, window=window, hop_length=win_length // 4)
                orig_signal = librosa.istft(clean_sig * phase, length=K, window=window, hop_length=win_length // 4)
                # Save as audio file on tensorboard
                self.writer.add_audio("reconstructed", rec_signal_hat, self.idx, sample_rate=22050)
                self.writer.add_audio("original", orig_signal, self.idx, sample_rate=22050)

                print("----------------------------------------------")
                print(f"Index {self.idx}")
                print(f"Mean train loss {self.mean_train_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean rec loss {self.rec_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean kl loss {self.kl_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean mag loss {self.mag_loss / conf.INTERVAL_TENSORBOARD}")
                print(f"Mean phase loss {self.phase_loss / conf.INTERVAL_TENSORBOARD}")
                print("----------------------------------------------")
                self.log_tensorboard()


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

    def apply(self, sfft_sig):
        """Apply model to a single sample"""
        denoise_sig, mean, log_var = self.autoencoder(sfft_sig)
        return denoise_sig, mean, log_var

    def train_sample(self, clean_sig, noised_sig, label):
        """Train a single sample"""
        # Move to device
        self.autoencoder.train()
        # Reshape batches
        clean_sig = clean_sig.swapaxes(1, 2).reshape(-1, 513)
        noised_sig = noised_sig.swapaxes(1, 2).reshape(-1, 513)
        clean_sig = clean_sig.to(self.device)

        # Apply model
        denoise_sig, mean, log_var = self.apply(clean_sig)


        # Calculate loss

        reconstruction_loss = self.loss(denoise_sig, clean_sig)

        # print(f"Rec loss: {reconstruction_loss.item()}, Path: {label}")

        # kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_loss = (torch.exp(2*log_var)  + mean ** 2 - log_var - 0.5).mean()
        loss = (1 - self.alpha) * reconstruction_loss + self.alpha * kl_loss
        self.mean_train_loss += loss.item()*conf.BATCH_SIZE
        self.rec_loss += reconstruction_loss.item()*conf.BATCH_SIZE
        self.kl_loss += kl_loss.item()*conf.BATCH_SIZE

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1)
        self.optimizer.step()
        return denoise_sig.detach(), mean.detach(), log_var.detach()


    def train(self):

        for epoch in range(10000):
            print(f"Epoch {epoch}")
            self.train_epoch()



if __name__ == "__main__":
    conf.INTERVAL_SAVE_MODEL = 50000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    conf.INTERVAL_TENSORBOARD = 100
    conf.INTERVAL_TENSORBOARD_PLOT = 1000
    conf.BATCH_SIZE = 1

    conf.LOAD_PATH = r"runs/fit/23-12-07-23H21_elegant_problem\models"
    conf.LOAD_IDX = 2900000


    trainer = Trainer(alpha=0.3)
    trainer.train()

# Command to run tensorboard
# tensorboard --logdir=runs/fit
# C:\ngrok\ngrok.exe http 6006