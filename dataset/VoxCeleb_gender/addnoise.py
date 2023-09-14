import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import math

file = "./males/0.wav"
#sr : frequenza di campionamento
#signal: campionamento * secondi ovvero tutti i campioni del segnale
clean_voice, sr = sf.read(file)
# sr is the frequency of the signal

# Root Mean Square ed è una misura della magnitudine di un segnale
RMS = math.sqrt(np.mean(clean_voice**2)) * 0.3
# RMS = 0.005

#Generazioen di rumore

#Viene creato un rumore gaussiano (rumore bianco) con una media di 0 e una deviazione standard di RMS.
#La lunghezza del rumore generato è uguale a quella del segnale.
noise = np.random.normal(0, RMS, clean_voice.shape[0])

#annulla il segnale nella prima e ultima parte (1/3 senza rumore,1/3 senza rumore,1/3 senza rumore)
noise[:int(len(noise)/3)] = 0
noise[len(noise) - int(len(noise)/3):] = 0

noised = clean_voice+noise

from scipy import signal


NFFT = 1028
mpl_specgram_window = plt.mlab.window_hanning(np.ones(NFFT))
# https://stackoverflow.com/questions/48598994/scipy-signal-spectrogram-compared-to-matplotlib-pyplot-specgram
f, t, Sxx = signal.spectrogram(noised, sr, detrend=False,
                               nfft=NFFT,
                               window=mpl_specgram_window,
                              )
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud', cmap='seismic')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


sf.write("noised.wav", noised, sr)
