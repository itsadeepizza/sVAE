
"""
1. Caricare la lista dei file audio (unprocessed)
2. scegliere un file audio
3. Campionare uno spezzone a caso di lunghezza K = 1024 -> s
4. Calcolare la STFT F = 513 di s
5. Calcolare la STFT del segnale + rumore

"""
import glob
import os
import soundfile as sf
import numpy     as np
from torch.utils.data import Dataset, DataLoader
import torch
import scipy
from config import selected_config as conf


class AudioDataset(Dataset):


    def find_audio_files( self ):
        """
        Find all audio files in the source path and store them in a list
        """
        if not os.path.exists(self.path):
            print("The specified path does not exist.")
            raise RuntimeError

        if not os.path.isdir(self.path):
            raise RuntimeError("The specified path is not a directory.")

        return glob.glob(self.path + "/**/*.wav", recursive=True)


    def __init__(self, path="dataset/unprocessed/VoxCeleb_gender/males"):
        self.path = path
        self._audio_files = self.find_audio_files()
        print(f"Found {len(self._audio_files)} audio files.")


    def __len__(self):
        return len(self._audio_files)


    def __getitem__(self, idx):
        K = 1024
        path_audio = self._audio_files[idx]
        signal = self.read_file(path_audio)
        # Crop the signal to the desired length
        if signal.shape[0] < K:
            # Pad with zeros if the signal is too short
            signal = np.pad(signal, (0, K - signal.shape[0]), mode='constant')
        start = np.random.randint(0, signal.shape[0] - K - 1)
        signal = signal[start:start + K]
        # Add noise
        noised_signal = self.add_noise_to_signal(signal)

        #Calculate STFT
        stft_signal = self.stft(signal)
        stft_noised_signal = stft_signal
        # stft_noised_signal = self.stft(noised_signal)

        # Convert to tensors
        stft_signal = torch.from_numpy(stft_signal).float()
        stft_noised_signal = torch.from_numpy(stft_noised_signal).float()

        # Normalize
        global_mean  = 2.1e-8
        global_var = 2.9e-6
        stft_signal = (stft_signal - global_mean) / np.sqrt(global_var)
        stft_noised_signal = (stft_noised_signal - global_mean) / np.sqrt(global_var)

        return stft_signal, stft_noised_signal


    def stft(self, signal):
        stft = scipy.signal.stft(signal, nperseg=1024)[2]
        return stft[:,0]


    def audio_files(self):
        if self._audio_files is None:
            self.find_audio_files()
        return self._audio_files


    def read_file(self, file_name) -> np.ndarray:
        """
        Load an audio file and return it as a numpy array
        """
        signal, _ = sf.read(file_name)
        return signal


    def add_noise_to_signal(self, signal: np.ndarray) -> np.ndarray:
        """
            trasforma il file con il nome source_file_name in un file con
            il noice e ne ritorna il path
        """
        max_RMS_sample = conf.RMS * np.sqrt(np.mean(signal ** 2))
        RMS_sample     = np.random.uniform(0, max_RMS_sample)
        noise          = np.random.normal(0, RMS_sample, signal.shape[0])
        noised         = signal + noise

        return noised


if __name__ == "__main__":

    dataset      = AudioDataset()
    dataloader   = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)