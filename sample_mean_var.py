from simple_loader import AudioDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from config import selected_config as conf
from simple_model import SimpleVAE
import matplotlib.pyplot as plt


vae = SimpleVAE()
vae.to(conf.DEVICE)
optimizer = torch.optim.Adam(vae.parameters(), lr=conf.LR)

# Load dataset
dataset = AudioDataset("dataset/unprocessed")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

loss_function = torch.nn.MSELoss()


tot_var = 0
tot_mean = 0
i = 0
n = 1000
for _, (clean_sig, noised_sig) in zip(range(n), dataloader):
        i +=1
        tot_mean += clean_sig.mean()
        tot_var += clean_sig.var()
        # Build a plot of distribution of clean_sig
        # clean_sig = torch.log(torch.abs(clean_sig) + 1)
        # plt.hist(clean_sig.flatten(), bins=100)
        plt.show()
        # input()

print(f"Mean: {tot_mean/n}")
print(f"Var: {tot_var/n}")