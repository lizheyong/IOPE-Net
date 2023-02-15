import sys
sys.path.append("..")
from .parts import *
import dataset


class Encode_Net(nn.Module):

    def __init__(self):
        super(Encode_Net, self).__init__()
        self.inc = Conv(1, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = DownRNN()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4

class Decode_Net1(nn.Module):

    def __init__(self):
        super(Decode_Net1, self).__init__()
        self.up1 = UpRNN()
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.out = OutConv(16, 1)

    def forward(self, x):
        x5 = self.up1(x)
        x6 = self.up2(x5)
        x7 = self.up3(x6)
        x7 = self.out(x7)
        return x7


class Decode_Net2(nn.Module):

    def __init__(self):
        super(Decode_Net2, self).__init__()
        self.up1 = UpRNN()
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.out = OutConv(16, 1)

    def forward(self, x):
        x5 = self.up1(x)
        x6 = self.up2(x5)
        x7 = self.up3(x6)
        x7 = self.out(x7)
        return x7

if __name__ == '__main__':
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load nets
    net0 = Encode_Net()
    net1 = Decode_Net1()
    net2 = Decode_Net2()
    # load nets to deivce
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)
    # load train_dataset
    HSI_dataset = dataset.HSI_Loader('../data/water_curves.npy')
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset, batch_size=1024, shuffle=True)
    # show a batch of train_data
    for curve, label in train_loader:
        len_of_this_batch = curve.shape[0]
        # curve and label should load to device
        curve = curve.reshape(len_of_this_batch, 1, -1).to(device=device, dtype=torch.float32)
        # label didn't add the dim
        label = label.to(device=device, dtype=torch.float32)
        # print a batch curve's shape [this_batch_size, 1, wavelength]
        print(curve.shape)
        encode_curve = net0(curve)
        # a [this_batch_size, 1, wavelength]
        a = net1(encode_curve)
        # bb [this_batch_size, 1, wavelength]
        bb = net2(encode_curve)
        print(f'{a.shape},a:{a[0, 0, :]},\n bb:{bb[0, 0, :]}')
        break
