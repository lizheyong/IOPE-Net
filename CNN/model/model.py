import torch.nn.functional as F
import torch
from .parts import * # 运行train时改为".parts"，model时候改为"parts"，路径问题
import sys
sys.path.append("..")
import dataset

class Encode_Net(nn.Module):

    def __init__(self, n_channels=1, n_classes=2):
        super(Encode_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = Conv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        return x3

class Decode_Net1(nn.Module):

    def __init__(self, n_channels=64, n_classes=1):
        super(Decode_Net1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.up1 = Up(n_channels, 32)
        self.up2 = Up(32, 16)
        self.out = OutConv(16, n_classes)

    def forward(self, x):

        x4 = self.up1(x)
        x5 = self.up2(x4)
        x6 = self.out(x5)
        return x6


class Decode_Net2(nn.Module):

    def __init__(self, n_channels=64, n_classes=1):
        super(Decode_Net2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.up1 = Up(n_channels, 32)
        self.up2 = Up(32, 16)
        self.out = OutConv(16, n_classes)

    def forward(self, x):

        x4 = self.up1(x)
        x5 = self.up2(x4)
        x6 = self.out(x5)
        return x6

if __name__ == '__main__':

    # net = IOPE_Net(n_channels=1, n_classes=2)
    # print(net)

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net0 = Encode_Net(n_channels=1, n_classes=64)
    net1 = Decode_Net1(n_channels=64, n_classes=1)
    net2 = Decode_Net2(n_channels=64, n_classes=1)

    # 将网络拷贝到deivce中
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)

    # 指定训练集地址，开始训练
    HSI_dataset = dataset.HSI_Loader('../data/all_curve.npy')
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=1024,
                                               shuffle=True)
    batch_size = 1024
    for curve, label in train_loader:
        # 将数据拷贝到device中
        curve = curve.reshape(batch_size, 1, -1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        break
    print(curve.shape)
        # break
        # 使用网络参数，输出预测结果
    encode_curve = net0(curve)
    a = net1(encode_curve)
    b = net2(encode_curve)
    print(curve[0,0,:])
    print(f'{a.shape},a:{a[0,0,:]},\n b:{b[0,0,:]}')
