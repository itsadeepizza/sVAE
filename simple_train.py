from simple_loader import AudioDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from config import selected_config as conf
from simple_model import SimpleVAE


vae = SimpleVAE()
vae.to(conf.DEVICE)
optimizer = torch.optim.Adam(vae.parameters(), lr=conf.LR)

# Load dataset
dataset = AudioDataset("dataset/unprocessed")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

loss_function = torch.nn.MSELoss()


tot_var = 0
tot_mean = 0
i = 0
for epoch in range(1000):
    for clean_sig, noised_sig in dataloader:
        i +=1
        device = conf.DEVICE
        clean_sig = torch.abs(clean_sig)
        noised_sig = torch.abs(noised_sig)

        clean_sig = clean_sig.to(device)
        # noised_sig = noised_sig.to(device)

        # Apply model (on clean signal)
        denoise_sig, mean, log_var = vae(clean_sig)

        # Calculate loss
        rec_loss = loss_function(denoise_sig, clean_sig)
        kl_loss = (torch.exp(log_var) ** 2 + mean ** 2 - log_var - 0.5).mean()

        alpha = 1
        loss = alpha * rec_loss + (1 - alpha) * kl_loss
        if i % 100 == 0:
            print(f"{i}: Tot. Loss: {loss.item():.4f} Rec. Loss: {rec_loss.item():.4f} KL Loss: {kl_loss.item():.4f}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()