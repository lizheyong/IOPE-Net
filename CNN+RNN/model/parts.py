import torch
import torch.nn as nn


class Conv(nn.Module):
    """(Conv => BN => ReLU )"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Pool => CONV"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        # If you use 'sequential' here, it will report an error, saying that 'forward' should be passed two parameters,
        # but I passed three, and after checking, 'sequential' cannot handle multiple inputs

    def forward(self, x):
        # Input x [1024, 64, 44] [batch_size, channel, seq_feature]
        batch_size = len(x) # Can't write 1024 directly, because the last batch is less than 1024
        x = torch.transpose(x, 1, 2) # Exchange dimension of x [1024, 44, 64] [batch_size, seq_len, feature]
        # Equivalent to 44 GRU units
        h0 = torch.randn(1, batch_size, 64).to(device='cuda') # h0 [num_layers, batch_size, hidden_size]
        output, hn = self.rnn(x, h0) # Output [1024, 44, 64] [batch_size, seq_len, hidden_size], hn [1, 1024, 64]
        return output

class UpRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

    def forward(self, x):
        # Input x [1024, 44, 64] [batch_size, seq_len, feature]
        batch_size = len(x)
        h0 = torch.randn(1, batch_size, 64).to(device='cuda') # h0 [num_layers, batch_size, hidden_size]
        output, hn = self.rnn(x, h0) # Output [1024, 44, 64] [batch_size, seq_len, hidden_size], hn [1, 1024, 64]
        output = torch.transpose(output, 1, 2) # Swap dimensions, back to the CNN with [1024, 64, 44]
        return output

class Up(nn.Module):
    """ConvTranspose => BN => Relu"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_conv(x)

class OutConv(nn.Module):
    """ConvTranspose"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outconv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.outconv(x)