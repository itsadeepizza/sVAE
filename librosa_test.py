import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np

file = "dataset/VoxCeleb_gender/males/0.wav"
y, sr = librosa.load(file)


# Add a gaussian noise in the half of the signal
RMS = np.sqrt(np.mean(y**2)) * 0.3
# y[0:int(len(y)/2)] = y[0:int(len(y)/2)] + np.random.normal(0, RMS, int(len(y)/2))

# Compute the spectrogram, preserving original length of the signal
n = len(y)
n_fft = 2048
y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)
D = librosa.stft(y_pad, n_fft=n_fft)

# Convert D to magnitude + phase
S, phase = librosa.magphase(D)
# Convert phase to radians
phase_angle = np.angle(phase)
# Convert S to decibel
ref = np.mean(S**2)
decib = librosa.amplitude_to_db(S, ref=ref)


# Plot the spectrogram
plt.figure(figsize=(12, 8))
librosa.display.specshow(decib,
                            y_axis='linear', x_axis='time', cmap='seismic')
# Maybe use log scale?
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')
plt.tight_layout()
plt.show()

# Invert the spectrogram
phase_hat = np.exp(1.j * phase_angle)
# convert decib to amplitude
S_hat = librosa.db_to_amplitude(decib, ref=ref)
# convert back to complex
D_hat = S_hat * phase_hat

y_hat = librosa.istft(D_hat, length=n)

# Export to file
soundfile.write("test_hat.wav", y_hat, sr)
soundfile.write("test.wav", y, sr)

# Get relative reconstruction error
print(f"Relative reconstruction error: {np.linalg.norm(y - y_hat) / np.linalg.norm(y)}")