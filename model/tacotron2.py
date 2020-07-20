import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .postnet import Postnet
from .speaker_encoder import SpeakerEncoder
from utils import get_mask_from_lengths


class Tacotron2(nn.Module):

    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels  # M
        self.n_fragment_mel_windows = hparams.n_fragment_mel_windows  # F

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.speaker_encoder = None
        if len(hparams.speaker_encoder) > 0:
            self.speaker_encoder = SpeakerEncoder(hparams)
        self.postnet = Postnet(hparams)

    def forward(self, inputs):
        text, text_lengths, mels, mel_lengths = inputs

        encoder_outputs = self.encoder(text, text_lengths)

        if self.speaker_encoder is not None:
            fragments = mels.unfold(1, self.n_fragment_mel_windows, self.n_fragment_mel_windows // 2).transpose(2, 3)
            fragment_counts = mel_lengths // (self.n_fragment_mel_windows // 2) - 1
            fragments = torch.cat([f[:fc] for f, fc in zip(fragments, fragment_counts)])
            speaker_embeddings = self.speaker_encoder.inference(fragments, fragment_counts.tolist())
            encoder_outputs = torch.cat([encoder_outputs,
                                         speaker_embeddings.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)], dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, text_lengths, mels)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if mel_lengths is not None:
            mask = ~get_mask_from_lengths(mel_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 2, 0)

            mel_outputs.data.masked_fill_(mask, 0.0)
            mel_outputs_postnet.data.masked_fill_(mask, 0.0)
            gate_outputs.data.masked_fill_(mask[:, :, 0], 1e3)  # gate energies
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs):
        text, mel = inputs

        encoder_outputs = self.encoder.inference(text)

        if self.speaker_encoder is not None:
            fragments = mel.unfold(1, self.n_fragment_mel_windows, self.n_fragment_mel_windows // 2).transpose(2, 3)[0]
            speaker_embeddings = self.speaker_encoder.inference(fragments, [fragments.size(0)])

            encoder_outputs = torch.cat([encoder_outputs,
                                         speaker_embeddings.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)], dim=2)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
