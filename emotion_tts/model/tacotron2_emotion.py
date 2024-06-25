import torch
import torch.nn as nn
import torch.nn.functional as F
from emotion_tts.model.layers import ConvNorm, LinearNorm

class Tacotron2Emotion(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Emotion, self).__init__()
        self.n_mel_channels = hparams['n_mel_channels']
        self.n_frames_per_step = 1  # currently only supports 1
        self.embedding = nn.Embedding(256, 512)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()
        self.emotion_embedding = nn.Embedding(len(hparams['EMOTIONS']), 512)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, emotions = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float()
        gate_padded = gate_padded.float()
        output_lengths = output_lengths.long()
        emotions = emotions.long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, emotions),
            (mel_padded, gate_padded))

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths, emotions = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        emotion_embed = self.emotion_embedding(emotions).unsqueeze(1)
        encoder_outputs = encoder_outputs + emotion_embed.expand_as(encoder_outputs)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        text_inputs, emotions = inputs
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        emotion_embed = self.emotion_embedding(emotions).unsqueeze(1)
        encoder_outputs = encoder_outputs + emotion_embed.expand_as(encoder_outputs)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    def parse_output(self, outputs, output_lengths=None):
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ConvNorm(512, 512, kernel_size=5)
        self.conv2 = ConvNorm(512, 512, kernel_size=5)
        self.conv3 = ConvNorm(512, 512, kernel_size=5)
        self.lstm = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        return outputs

    def inference(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.n_mel_channels = 80
        self.n_frames_per_step = 1
        self.encoder_embedding_dim = 512
        self.attention_rnn_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim)

        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            self.attention_dim, self.attention_location_n_filters,
            self.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            self.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step)

        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def forward(self, memory, decoder_inputs, memory_lengths):
        # ... (implement forward pass)
        pass

    def inference(self, memory):
        # ... (implement inference)
        pass

class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512)))
        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512)))
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80)))

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        # ... (implement attention mechanism)
        pass

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        # ... (implement forward pass)
        pass

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask