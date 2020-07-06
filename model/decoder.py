import torch
from torch import nn
from torch.nn import functional as F
from model.layers import Linear
from .attention import Attention
from utils import module_device, get_mask_from_lengths


class Decoder(nn.Module):

    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        if hparams.speaker_encoder:
            self.encoder_embedding_dim += hparams.speaker_encoder_dim
        self.prenet_dim = hparams.prenet_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.gate_threshold = hparams.gate_threshold
        self.max_decoder_steps = hparams.max_decoder_steps
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.dtype = torch.float16 if hparams.fp16_run else torch.float32

        self.prenet = Prenet(
            hparams.n_mel_channels,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim)

        self.attention_layer = Attention(
            self.encoder_embedding_dim, hparams.attention_rnn_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim)

        self.linear_projection = Linear(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels)

        self.gate_layer = Linear(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def forward(self, memory, memory_lengths, decoder_inputs):
        """
        :param memory: B, T, E
        :param memory_lengths: B
        :param decoder_inputs: B, S, M
        :return: mel_outputs: B, S, M; gate_outputs: B, S; alignments: B, S
        """
        B, T, _ = memory.size()
        _, S, _ = decoder_inputs.size()
        self.initialize_decoder_states(B, T)
        self.memory = memory
        self.processed_memory = self.attention_layer.forward_memory(memory)
        self.mask = ~get_mask_from_lengths(memory_lengths).to(device=module_device(self))

        decoder_input = self.get_go_frame(B).unsqueeze(0)  # 1, B, M
        decoder_inputs = decoder_inputs.transpose(0, 1)  # S, B, M
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)  # S + 1, B, M
        decoder_inputs = self.prenet(decoder_inputs)  # S + 1, B, D

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < S:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [alignment]

        alignments = torch.stack(alignments).transpose(0, 1)  # B, S, T
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()  # B, S
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()  # B, S, M

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        B, T, _ = memory.size()
        self.initialize_decoder_states(B, T)
        self.memory = memory
        self.processed_memory = self.attention_layer.forward_memory(memory)
        self.mask = None

        decoder_input = self.get_go_frame(B)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input.unsqueeze(0))[0]
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print('Warning! Reached max decoder steps')
                break

            decoder_input = mel_output

        alignments = torch.stack(alignments).transpose(0, 1)  # B, S, T
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()  # B, S
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()  # B, S, M

        return mel_outputs, gate_outputs, alignments

    def get_go_frame(self, B):
        return torch.zeros(B, self.n_mel_channels, device=module_device(self), dtype=self.dtype)

    def initialize_decoder_states(self, B, T):
        self.attention_hidden = torch.zeros(B, self.decoder_rnn_dim, device=module_device(self), dtype=self.dtype)
        self.attention_cell = torch.zeros(B, self.decoder_rnn_dim, device=module_device(self), dtype=self.dtype)

        self.decoder_hidden = torch.zeros(B, self.decoder_rnn_dim, device=module_device(self), dtype=self.dtype)
        self.decoder_cell = torch.zeros(B, self.decoder_rnn_dim, device=module_device(self), dtype=self.dtype)

        self.attention_weights = torch.zeros(B, T, device=module_device(self), dtype=self.dtype)
        self.attention_weights_cum = torch.zeros(B, T, device=module_device(self), dtype=self.dtype)
        self.attention_context = torch.zeros(B, self.encoder_embedding_dim, device=module_device(self), dtype=self.dtype)

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)  # B, M + E
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), -1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights


class Prenet(nn.Module):

    def __init__(self, in_dim, out_dims):
        super(Prenet, self).__init__()

        self.layers = nn.ModuleList([
            Linear(in_dim, out_dim, bias=False)
            for in_dim, out_dim
            in zip([in_dim] + out_dims[:-1], out_dims)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), 0.5, training=True)
        return x

