import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np
import cv2

file = "dataset/VoxCeleb_gender/males/0.wav"
y, sr = librosa.load(file)


# Add a gaussian noise in the half of the signal
RMS = np.sqrt(np.mean(y**2)) * 0.3
# y[0:int(len(y)/2)] = y[0:int(len(y)/2)] + np.random.normal(0, RMS, int(len(y)/2))

# Compute the spectrogram, preserving original length of the signal
n     = len(y)
n_fft = 512 # in speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz
y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)
D     = librosa.stft(y_pad, n_fft=n_fft)

# Convert D to magnitude + phase
S, phase = librosa.magphase(D)
# Convert phase to radians
phase_angle = np.angle(phase)
# Convert S to decibel
ref = np.mean(S**2)
decib = librosa.amplitude_to_db(S, ref=ref)



# # compress decib using opencv
# shape = decib.shape
# small_decib = cv2.resize(decib, (decib.shape[1]//2, decib.shape[0]//2), interpolation=cv2.INTER_AREA)
#
# # Resize decib to original size
# decib_hat = cv2.resize(small_decib, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
#
# # Print relative error for decib
# print(f"Relative error for decib: {np.linalg.norm(decib - decib_hat) / np.linalg.norm(decib)}")

decib_hat = decib

# Plot the spectrogram
plt.figure(figsize=(12, 8))
librosa.display.specshow(decib_hat,
                            y_axis='linear', x_axis='time', cmap='seismic')
# Maybe use log scale?
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')
plt.tight_layout()
plt.show()

# Invert the spectrogram
phase_hat = np.exp(1.j * phase_angle)
# convert decib to amplitude
S_hat = librosa.db_to_amplitude(decib_hat, ref=ref)
# convert back to complex
D_hat = S_hat * phase_hat
# D_hat = S_hat

y_hat = librosa.istft(D_hat, length=n)

# Export to file
soundfile.write("test_hat.wav", y_hat, sr)
soundfile.write("test.wav", y, sr)

# Get relative reconstruction error
print(f"Relative reconstruction error: {np.linalg.norm(y - y_hat) / np.linalg.norm(y)}")