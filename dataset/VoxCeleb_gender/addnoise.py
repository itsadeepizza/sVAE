
import soundfile as sf
import numpy as np
import math

file = ""
signal, sr = sf.read(file)

#RMS = math.sqrt(np.mean(signal**2))
RMS = 0.005

noise = np.random.normal(0, RMS, signal.shape[0])
noise[:int(len(noise)/3)] = 0
noise[len(noise) - int(len(noise)/3):] = 0

noised = signal+noise

sf.write("noised.wav", noised, sr)
