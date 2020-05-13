from torch import nn
from math import sqrt


class Embedding(nn.Module):

    def __init__(self, hparams):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        val = 1 / sqrt(hparams.symbols_embedding_dim)
        nn.init.uniform_(self.embedding.weight, -val, val)

    def forward(self, inputs):
        return self.embedding(inputs)

