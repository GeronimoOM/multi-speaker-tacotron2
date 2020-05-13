import torch
from torch import nn
from torch.nn import functional as F
from model.layers import Conv1d


class Postnet(nn.Module):

    def __init__(self, hparams):
        super(Postnet, self).__init__()

        self.convolutions = nn.ModuleList()
        n_conv = hparams.postnet_n_convolutions
        for i in range(n_conv):
            in_dim = hparams.n_mel_channels if i == 0 else hparams.postnet_embedding_dim
            out_dim = hparams.postnet_embedding_dim if i < n_conv - 1 else hparams.n_mel_channels
            self.convolutions.append(nn.Sequential(
                Conv1d(in_dim, out_dim,
                       kernel_size=hparams.postnet_kernel_size,
                       w_init_gain='tanh' if i < n_conv - 1 else 'linear'),
                nn.BatchNorm1d(out_dim))
            )

    def forward(self, x):
        for i, conv in enumerate(self.convolutions):
            x = conv(x)
            if i < len(self.convolutions) - 1:
                x = torch.tanh(x)
            x = F.dropout(x, 0.5, self.training)

        return x
