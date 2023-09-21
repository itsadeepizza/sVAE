import numpy     as np
import librosa
import glob
import soundfile as sf
import os
from tqdm import tqdm
import re

from typing import Sequence

class Dataset_Builder():

    def __init__( self,
                  source_path,
                  output_path,
                  hop_length = 160,
                  n_fft      = 512,
                  spec_len   = 250,
                  win_length = 400,
                  RMS        = 0.3 ):



        self.source_path  = source_path
        self.output_path  = output_path
        self.hop_length   = hop_length
        self.n_fft        = n_fft
        self.spec_len     = spec_len
        self.win_length   = win_length
        self.RMS          = RMS
        self._audio_files = None

    @property
    def audio_files(self):
        if self._audio_files is None:
            self.find_audio_files()
        return self._audio_files

    def add_audio_files(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".mp3") or file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, path)
                    self.audio_files.append(relative_path)

    def get_audio_files(self):
        return self.audio_files


    def convert_source_path_to_output_path(self, source_path):
        output_path = re.sub(self.source_path, self.output_path, source_path)
        output_path = output_path.replace(".wav", ".npz")
        return output_path

    def check_file_exists(file_path, target_path):
        target_file_path = os.path.join(target_path, os.path.basename(file_path))
        return os.path.exists(target_file_path)

    def add_noise_to_signal(self, signal: np.ndarray) -> np.ndarray:
        """
            trasforma il file con il nome source_file_name in un file con
            il noice e ne ritorna il path
        """
        max_RMS_sample = self.RMS * np.sqrt(np.mean(signal ** 2))
        RMS_sample     = np.random.uniform(0, max_RMS_sample)
        noise          = np.random.normal(0, RMS_sample, signal.shape[0])
        noised         = signal + noise

        return noised


    def read_file(self, file_name) -> np.ndarray:
        """
        Load an audio file and return it as a numpy array
        """
        signal, _ = sf.read(file_name)
        return signal


    def process_file(self, file_name):
        """
        Generete spectrogram of the noised and clean audio file and save it as a pickle file
        """
        signal = self.read_file(file_name)
        noised_signal = self.add_noise_to_signal(signal)

        clean_Tmag, clean_phase = self.make_spectrogram(signal)
        noised_Tmag, noised_phase = self.make_spectrogram(noised_signal)

        # Save the spectrograms
        # TODO: save the spectrograms as pickle files
        out_file_name = self.convert_source_path_to_output_path(file_name)
        base_dir = os.path.dirname(out_file_name)
        os.makedirs(base_dir, exist_ok=True)
        np.savez(out_file_name,
                    clean_Tmag=clean_Tmag,
                    clean_phase=clean_phase,
                    noised_Tmag=noised_Tmag,
                    noised_phase=noised_phase)



    def process_all_files(self):
        """
            metodo che prende tutti i file sorgenti e li trasforma in file con il noice.
            da lanciare una tantum (per preparare il dataset)
        """
        ###############################################

        for file_name in tqdm(self.audio_files):
            try:
                print(f"Processing {file_name}...", end='')
                if not self.file_is_already_processed(file_name):
                    self.process_file(file_name)
                    # Il metodo ritorna True
                    # Procedi con il prossimo file
                print("Done.")
            except Exception as e:
                print("Eccezione generata per il file {}: {}".format(file_name, str(e)))
        ##################################################


    def file_is_already_processed(self, source_file_name):
        """
            controlla se il relativo source_file_name ha gia' il corrispettivo file con la noice aggiunta
        """
        output_file_name = self.convert_source_path_to_output_path(source_file_name)

        return os.path.isfile(output_file_name)

    def make_spectrogram( self, signal: np.ndarray ) -> (np.ndarray, np.ndarray):

        """
            Takse an audio signal and returns its spectrogram as a (mag, phase) tuple
        """
        linear = librosa.stft( signal, n_fft = self.n_fft, win_length= self.win_length, hop_length=self.hop_length )
        mag, phase = librosa.magphase(linear)
        return (mag.T, phase)



    def find_audio_files( self ) -> None:
        """
        Find all audio files in the source path and store them in a list
        """
        if not os.path.exists(self.source_path):
            print("The specified path does not exist.")
            raise RuntimeError

        if not os.path.isdir(self.source_path):
            raise RuntimeError("The specified path is not a directory.")
        self._audio_files = glob.glob(self.source_path + "/**/*.wav", recursive=True)
        print(f"Found {len(self._audio_files)} audio files.")

if __name__ == "__main__":
    builder = Dataset_Builder(source_path="./dataset/unprocessed", output_path="./dataset/spectrograms")
    builder.process_all_files()