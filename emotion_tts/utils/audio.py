import librosa
import numpy as np
import torch
from emotion_tts.config import HPARAMS

def load_wav_to_torch(full_path):
    sampling_rate = HPARAMS['sampling_rate']
    data, _ = librosa.load(full_path, sr=sampling_rate)
    return torch.FloatTensor(data.astype(np.float32))

def mel_spectrogram_to_wav(mel_spectrogram, n_iter=10, sr=22050, n_mels=80, n_fft=2048, hop_length=512):
    # Convert mel spectrogram to magnitude spectrogram
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mag_spec = np.dot(mel_basis.T, mel_spectrogram)

    # Generate phase information using Griffin-Lim algorithm
    phases = np.exp(2j * np.pi * np.random.rand(*mag_spec.shape))
    stft = mag_spec * phases

    for i in range(n_iter):
        y = librosa.istft(stft, hop_length=hop_length)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        phases = np.exp(1j * np.angle(stft))
        stft = mag_spec * phases

    # Convert complex spectrogram to waveform
    waveform = librosa.istft(stft, hop_length=hop_length)
    
    return waveform
