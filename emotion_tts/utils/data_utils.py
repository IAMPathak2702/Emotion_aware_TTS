import torch
import numpy as np
from torch.utils.data import Dataset
from emotion_tts.utils.text_processing import text_to_sequence

class TextMelLoader(Dataset):
    def __init__(self, file_path, hparams):
        self.file_path = file_path
        self.hparams = hparams
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    data.append(parts)
        return data

    def get_mel(self, filename):
        # Placeholder: You need to implement mel spectrogram extraction
        return np.random.rand(80, 100)  # Random mel for now

    def __getitem__(self, index):
        file_path, text, emotion = self.data[index]
        text = torch.LongTensor(text_to_sequence(text, ['english_cleaners']))
        mel = torch.FloatTensor(self.get_mel(file_path))
        emotion = torch.LongTensor([self.hparams['EMOTIONS'].index(emotion)])
        return (text, mel, emotion)

    def __len__(self):
        return len(self.data)

class TextMelCollate:
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        emotions = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            emotions[i] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, emotions