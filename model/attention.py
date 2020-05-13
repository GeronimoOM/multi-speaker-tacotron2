import torch
from torch import nn
from torch.nn import functional as F
from model.layers import Conv1d, Linear


class Attention(nn.Module):

    def __init__(self, embedding_dim, attention_rnn_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.memory_layer = Linear(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.query_layer = Linear(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.location_layer = Location(attention_location_n_filters, attention_location_kernel_size, attention_dim)
        self.linear = Linear(attention_dim, 1, bias=False)

    def forward_memory(self, memory):
        """
        :param memory: B, T, E
        :return: processed_memory: B, T, A
        """
        return self.memory_layer(memory)

    def forward(self, memory, processed_memory,
                query, attention_weights_cat, mask):
        """
        :param memory: B, T, E
        :param processed_memory: B, T, A
        :param query: D
        :param attention_weights_cat: B, 2, T
        :param mask: B, T
        :return: attention_context: B; attention_weights: B, T
        """
        processed_query = self.query_layer(query.unsqueeze(1))  # B, 1, A
        processed_attention_weights = self.location_layer(attention_weights_cat)  # B, T, A

        alignment = self.linear(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        alignment = alignment.squeeze(-1)  # B, T

        if mask is not None:
            alignment.data.masked_fill_(mask, -float('inf'))

        attention_weights = F.softmax(alignment, dim=1)  # B, T
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)  # B

        return attention_context, attention_weights


class Location(nn.Module):

    def __init__(self, attention_location_n_filters, attention_location_kernel_size, attention_dim):
        super(Location, self).__init__()
        self.location_conv = Conv1d(2, attention_location_n_filters,
                                    kernel_size=attention_location_kernel_size,
                                    bias=False)
        self.location_dense = Linear(attention_location_n_filters, attention_dim,
                                     bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        """
        :param attention_weights_cat: B, 2, T
        :return: processed_attention: B, T, A
        """
        processed_attention = self.location_conv(attention_weights_cat)  # B, L, T
        processed_attention = processed_attention.transpose(1, 2)  # B, T, L
        processed_attention = self.location_dense(processed_attention)  # B, T, A
        return processed_attention
