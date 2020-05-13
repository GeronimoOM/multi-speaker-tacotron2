from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .postnet import Postnet
from utils import get_mask_from_lengths


class Tacotron2(nn.Module):

    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels  # M

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def forward(self, inputs):
        text, text_lengths, mels, mels_lengths = inputs

        encoder_outputs = self.encoder(text, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, text_lengths, mels)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if mels_lengths is not None:
            mask = ~get_mask_from_lengths(mels_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_outputs.data.masked_fill_(mask, 0.0)
            mel_outputs_postnet.data.masked_fill_(mask, 0.0)
            gate_outputs.data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
