from torch import nn
from torch.nn import functional as F
from model.layers import Conv1d, Embedding


class Encoder(nn.Module):

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        self.embedding = Embedding(hparams)

        self.convolutions = nn.ModuleList()
        for i in range(hparams.encoder_n_convolutions):
            self.convolutions.append(nn.Sequential(
                Conv1d(hparams.symbols_embedding_dim if i == 0 else hparams.encoder_embedding_dim,
                       hparams.encoder_embedding_dim,
                       kernel_size=hparams.encoder_kernel_size,
                       w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim)))

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2),
                            batch_first=True,
                            bidirectional=True)

    def forward(self, text, text_lengths):
        """
        :param text: B, T
        :param text_lengths: B
        :return: encodings B, T, E
        """

        x = self.embedding(text).transpose(1, 2)  # B, Em, T

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)  # B, E, T

        x = x.transpose(1, 2)  # B, T, E

        if text_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, text_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # B, T, E

        if text_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def inference(self, text):
        return self.forward(text, None)
