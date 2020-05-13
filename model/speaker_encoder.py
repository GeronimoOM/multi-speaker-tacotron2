from torch import nn
from torch.nn import functional as F


class SpeakerEncoder(nn.Module):

    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(hparams.speaker_encoder_n_layers):
            self.layers.append(nn.Sequential(
                nn.LSTM(hparams.n_mel_channels if i == 0 else hparams.speaker_encoder_dim,
                        hparams.speaker_encoder_rnn_dim,
                        batch_first=True),
                nn.Linear(hparams.speaker_encoder_rnn_dim, hparams.speaker_encoder_dim)
            ))

    def forward(self, mel):
        """
        :param mel: B, M, T
        :return: speaker embeddings B, S
        """

        x = mel.transpose(1, 2)  # B, T, M
        for layer in self.layers:
            x, _ = layer[0](x)
            x = layer[1](x)  # B, T, S

        # pick last outputs
        x = F.normalize(x[:, -1], p=2)

        return x
