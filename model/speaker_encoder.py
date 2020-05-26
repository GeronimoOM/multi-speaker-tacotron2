import torch
from torch import nn
from torch.nn import functional as F


class SpeakerEncoder(nn.Module):

    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(hparams.speaker_encoder_n_layers):
            lstm = nn.LSTM(hparams.n_mel_channels if i == 0 else hparams.speaker_encoder_dim,
                           hparams.speaker_encoder_rnn_dim,
                           batch_first=True)
            lstm.flatten_parameters()

            self.layers.append(
                nn.Sequential(lstm, nn.Linear(hparams.speaker_encoder_rnn_dim, hparams.speaker_encoder_dim)))

    def forward(self, mels):
        """
        :param mels: B, T, M
        :return: speaker embeddings B, S
        """

        x = mels.contiguous()
        for layer in self.layers:
            x, _ = layer[0](x)
            x = layer[1](x)  # B, T, S

        # pick last outputs
        x = F.normalize(x[:, -1], p=2)

        return x

    def inference(self, mels, mel_counts):
        embeddings = self.forward(mels)
        return F.normalize(torch.stack([f.mean(axis=0) for f in embeddings.split(mel_counts)]), p=2)
