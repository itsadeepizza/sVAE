import glob
import numpy as np
import torch


from config           import selected_config     as conf
from torch.utils.data import Dataset, DataLoader


def load_dataset(spectrograms_path):
    spectrograms = glob.glob(spectrograms_path + "/**/*.npz", recursive=True)
    print(f"Found {len(spectrograms)} spectrograms.")
    return spectrograms


class AudioDataset(Dataset):

    def __init__(self, spectrograms_list, length=250):
        self.spectrograms_list = spectrograms_list
        self.length = length

    def __len__(self):
        return len(self.spectrograms_list)

    def __getitem__(self, idx):

        npz_path      = self.spectrograms_list[idx]
        npz_dict      = np.load(npz_path)
        clean_Tmag    = npz_dict["clean_Tmag"]
        clean_Tphase  = npz_dict["clean_Tphase"]
        noised_Tmag   = npz_dict["noised_Tmag"]
        noised_Tphase = npz_dict["noised_Tphase"]

        # Convert to full precision pytorch tensors
        clean_Tmag    = torch.from_numpy(clean_Tmag).float()
        clean_Tphase  = torch.from_numpy(clean_Tphase).float()
        noised_Tmag   = torch.from_numpy(noised_Tmag).float()
        noised_Tphase = torch.from_numpy(noised_Tphase).float()

        # Crop the spectrograms to the desired length
        # If the spectrograms are shorter than the desired length, pad them with zeros
        if min(clean_Tmag.shape[0], clean_Tphase.shape[0], noised_Tmag.shape[0], noised_Tphase.shape[0]) < self.length:

            clean_Tmag    = torch.cat((clean_Tmag, torch.zeros(self.length - clean_Tmag.shape[0], clean_Tmag.shape[1])), dim=0)
            clean_Tphase  = torch.cat((clean_Tphase, torch.zeros(self.length - clean_Tphase.shape[0], clean_Tphase.shape[1])), dim=0)
            noised_Tmag   = torch.cat((noised_Tmag, torch.zeros(self.length - noised_Tmag.shape[0], noised_Tmag.shape[1])), dim=0)
            noised_Tphase = torch.cat((noised_Tphase, torch.zeros(self.length - noised_Tphase.shape[0], noised_Tphase.shape[1])), dim=0)


        # Take a random starting point
        start         = np.random.randint(0, clean_Tmag.shape[0] - self.length - 1)
        clean_Tmag    = clean_Tmag[start:start + self.length, :]
        clean_Tphase  = clean_Tphase[start:start + self.length, :]
        noised_Tmag   = noised_Tmag[start:start + self.length, :]
        noised_Tphase = noised_Tphase[start:start + self.length, :]


        return clean_Tmag, clean_Tphase, noised_Tmag, noised_Tphase





if __name__ == "__main__":


    spectrograms = load_dataset('dataset/spectrograms')
    dataset      = AudioDataset(spectrograms, length=conf.SIGNAL_LENGTH)
    dataloader   = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for clean_Tmag, clean_phase, noised_Tmag, noised_phase in dataloader:
        print(clean_Tmag.shape)
        print(clean_phase.shape)
        print(noised_Tmag.shape)
        print(noised_phase.shape)


