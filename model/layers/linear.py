from torch import nn


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear(x)