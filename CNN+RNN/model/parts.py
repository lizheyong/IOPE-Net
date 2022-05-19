import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #这里用sequential的话就会报错，说forward要传入两个参数，但是我传入了三个，查了下，sequential不能处理多输入

    def forward(self, x):
        # 输入的x [1024, 64, 44] [batch_size, channel, seq_feature]
        batch_size = len(x) # 不能直接写1024，因为最后一个batch不足1024
        x = torch.transpose(x, 1, 2) # 交换维度的x [1024, 44, 64] [batch_size, seq_len, feature]
        # 相当于44个GRU单元
        h0 = torch.randn(1, batch_size, 64).to(device='cuda') # h0 [num_layers, batch_size, hidden_size]
        output, hn = self.rnn(x, h0) # output [1024, 44, 64] [batch_size, seq_len, hidden_size], hn [1, 1024, 64]
        return output

class UpRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

    def forward(self, x):
        # 这时候输入的x [1024, 44, 64] [batch_size, seq_len, feature]
        batch_size = len(x) # 不能直接写1024，因为最后一个batch不足1024
        h0 = torch.randn(1, batch_size, 64).to(device='cuda') # h0 [num_layers, batch_size, hidden_size]
        output, hn = self.rnn(x, h0) # output [1024, 44, 64] [batch_size, seq_len, hidden_size], hn [1, 1024, 64]
        output = torch.transpose(output, 1, 2) # 交换维度,变回CNN用的 [1024, 64, 44]
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
