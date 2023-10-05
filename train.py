from loader import load_dataset, AudioDataset
from torch.utils.data import DataLoader
from config import selected_config as conf
from model import VariationalAutoEncoder

if __name__ == "__main__":

    spectrograms = load_dataset('dataset/spectrograms')
    dataset = AudioDataset(spectrograms, length=conf.SIGNAL_LENGTH)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    autoencoder = VariationalAutoEncoder()

    for clean_Tmag, clean_Tphase, noised_Tmag, noised_Tphase in dataloader:
        Tmag_hat, Tphase_hat = autoencoder(noised_Tmag, noised_Tphase)
        loss = Tmag_hat - clean_Tmag
        print(loss)