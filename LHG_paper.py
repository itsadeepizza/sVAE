# https://arxiv.org/pdf/1902.01605.pdf
import librosa
import scipy.signal.windows
import soundfile
import matplotlib.pyplot as plt
import numpy             as np

file = "dataset/unprocessed/VoxCeleb_gender/males/0.wav"
y, sr = librosa.load(file)
# Print sampling rate
print(f"Sampling rate is {sr}")
# Resample the file
y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)

print(y.shape, y_16k.shape)
# (121364,) (88065,)
# OK 88065 = 121364 / 22060 * 16000

# The STFT is computed using a 64-ms sine
# window1 (i.e. F = 513) with 75%-overlap

# At 16 KHz, 64 ms = 1024 points
win_length = 1024
window = scipy.signal.windows.cosine(win_length)
S = np.abs(librosa.stft(y,
                        n_fft=win_length,
                        win_length=win_length,
                        hop_length=win_length // 4,
                        window=window
                        ))
# Take squared module (power)
S = np.abs(S)**2


print(S.shape)

