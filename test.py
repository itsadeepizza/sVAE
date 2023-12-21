from simpletrain2 import Trainer
from config import selected_config as conf
import torch
import numpy as np
import librosa
import scipy
import soundfile as sf


K = 1024 * 25
win_length = 1024

sound_path = r"dataset/unprocessed/VoxCeleb_gender/females/60.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# conf.LOAD_PATH = r"runs/fit/23-12-07-23H21_elegant_problem\models"
conf.LOAD_PATH = r"runs/fit/23-12-10-00H54_red_government\models"
conf.LOAD_IDX = 13000000

signal, sr = librosa.load(sound_path)

pad_length = K - (len(signal) % K)
signal = np.pad(signal, (0, pad_length), mode='constant')
signal = signal.reshape(-1, K)

trainer = Trainer()
# Eval mode
trainer.autoencoder.eval()

processed_signal = []

for signal_chunk in signal:
    signal_chunk = signal_chunk/np.sqrt(np.mean(signal_chunk ** 2))

    #Calculate STFT
    window = scipy.signal.windows.cosine(win_length)
    stft_signal = librosa.stft(signal_chunk, n_fft=win_length, win_length=win_length, hop_length=win_length // 4, window=window)
    stft_signal, phase = librosa.magphase(stft_signal)

    stft_signal = np.abs(stft_signal) ** 2
#     Convert to tensors
    stft_signal = torch.from_numpy(stft_signal).float()
    stft_signal = stft_signal.to(DEVICE)
    # Normalize
    global_var = 68210016 * 0.061
    stft_signal = stft_signal / np.sqrt(global_var)


    # Apply model
    stft_signal = stft_signal.swapaxes(0, 1)
    with torch.no_grad():
        denoise_sig, mean, log_var = trainer.apply(stft_signal)
        denoise_sig = stft_signal

    # Invert stft (seems to work flawlessy)
    denoise_sig = denoise_sig.cpu().numpy()
    denoise_sig = denoise_sig.swapaxes(1, 0)
    signal_hat = librosa.istft(denoise_sig*phase, length = K, window=window, hop_length=win_length // 4)
    processed_signal.append(signal_hat)

processed_signal = np.concatenate(processed_signal)
sf.write("not_processed.wav", processed_signal, int(sr))








