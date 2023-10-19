import numpy as np
import librosa

def spec2sound(Tmag, Tphase, n=None):
    """
    Convert magnitude and phase to sound
    :param Tmag: Transpose of the magnitude spectrogram
    :param Tphase: Transpose of the phase spectrogram (converted to radiant)
    :return: sound signal
    """
    # Convert angle
    Tphase_as_complex = np.exp(1.j * Tphase) # complex value = e^(iθ)
    phase = Tphase_as_complex.T

   # Convert magnitude
    mag = Tmag.T
    # Convert to complex
    # ref = ?
    # mag = librosa.db_to_amplitude(mag, ref=ref)

    # Convert to complex
    D = mag * phase
    # Inverse spectrogram
    if n is None:
        n = int(mag.shape[1] * 121364/951)
    y = librosa.istft(D, length=n, win_length=400, hop_length=160, n_fft=512)
    return y
