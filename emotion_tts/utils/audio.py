import librosa
import numpy as np
import torch
from emotion_tts.config import HPARAMS

def load_wav_to_torch(full_path):
    sampling_rate = HPARAMS['sampling_rate']
    data, _ = librosa.load(full_path, sr=sampling_rate)
    return torch.FloatTensor(data.astype(np.float32))

def mel_spectrogram_to_wav(mel_spectrogram):
    # This is a placeholder function. You need to implement or use a vocoder
    # to convert mel spectrograms to waveforms.
    raise NotImplementedError("Implement a vocoder to convert mel spectrograms to waveforms.")