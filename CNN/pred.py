import matplotlib.pyplot as plt
from model.model import Encode_Net, Decode_Net1, Decode_Net2
from dataset import HSI_Loader
import numpy as np
from torch import optim
import torch.nn as nn
import torch
import sys
import math
from log import Logger
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
torch.set_printoptions(precision=16, threshold=float('inf'))
sys.stdout = Logger()


def pred_net(net0, net1, net2, device, npyfile, batch_size=1024):
    # 加载训练集
    HSI_dataset = HSI_Loader(npyfile)
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    criterion = nn.MSELoss()
    # 测试模式
    # 加载模型参数
    net0.load_state_dict(torch.load('best_model_net0.pth', map_location=device))# 加载模型参数
    net1.load_state_dict(torch.load('best_model_net1.pth', map_location=device))# 加载模型参数
    net2.load_state_dict(torch.load('best_model_net2.pth', map_location=device))
    net0.eval()
    net1.eval()
    net2.eval()
    pred_loss = 0
    n = 0
    for curve, label in train_loader:
        # 将数据拷贝到device中
        curve = curve.reshape(len(curve), 1, -1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        encode_out = net0(curve)
        a = net1(encode_out)
        b = net2(encode_out)
        u = b / (a+b)
        r = (0.084 + 0.170 * u) * u
        r = torch.squeeze(r)
        # 计算loss
        pltcurve = 1 # 要看哪条曲线

        print(f'r:{r[pltcurve,:]}')
        print(f'a:{a[pltcurve,0,:]}')
        print(f'b:{b[pltcurve,0,:]}')
        print(f'label:{label[pltcurve,:]}')

        def SALoss(r, label):
            # 下面得到每行的二范数，也是再用哈达玛积相乘
            r_l2_norm = torch.norm(r, p=2, dim=1)  # [1024]
            label_l2_norm = torch.norm(label, p=2, dim=1)  # [1024]
            # r*label,对应元素乘,hadamard积,[1024,176],然后，每行求和torch.sum(r*label,dim=1)
            # 这样得到的是“向量r与向量label的内积”
            SALoss = torch.sum(
                torch.acos(torch.sum(r * label, dim=1) / (r_l2_norm * label_l2_norm))
            )  # acos括号内为[1024]
            SALoss /= math.pi * len(r)  # 除以pi归一化到[0,1]，除以batch_size平均一下
            return SALoss

        # 计算loss
        loss = 1e7 * criterion(r, label) + SALoss(r, label)
        # 计算相对误差百分比REP(relative error percentage)
        REP = torch.abs(r-label)/label # REP [1024,176]
        sum_REP = torch.where(torch.isinf(torch.sum(REP, 1)), torch.full_like(torch.sum(REP, 1), 1), torch.sum(REP, 1)) # inf换为1
        mean_REP = torch.mean(sum_REP)
        print(f'loss:{loss.item()}')
        print(f'REP:{torch.sum(REP[pltcurve,:]).item()}')
        print(f'mean_REP:{mean_REP.item()}')

        # 画图
        wavelength = np.load(r"C:\Users\423\Desktop\铁测试\wavelength.npy")
        # 重构曲线和真实曲线
        plt.figure()

        plt.plot(wavelength, gaussian_filter1d(torch.squeeze(r[pltcurve,:]).detach().cpu().numpy(),sigma=5),
                 label='reconstruct', color='r', marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(torch.squeeze(label[pltcurve,:]).detach().cpu().numpy(),sigma=5),
                 label='real', color='b', marker='o', markersize=3)
        plt.xlabel('band')
        plt.ylabel('reflect value')
        plt.legend()
        # a和b_b
        plt.figure()
        a_smoothed = gaussian_filter1d(torch.squeeze(a[pltcurve,:]).detach().cpu().numpy(), sigma=5)
        plt.plot(wavelength, a_smoothed,
                 label='a', color='r', marker='o', markersize=3)
        b_smoothed = gaussian_filter1d(torch.squeeze(b[pltcurve,:]).detach().cpu().numpy(), sigma=5)
        plt.plot(wavelength, b_smoothed,
                 label='bb', color='b', marker='o', markersize=3)
        plt.xlabel('band')
        plt.ylabel('value')
        plt.legend()

        plt.show()
        break




if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net0 = Encode_Net(n_channels=1, n_classes=64)
    net1 = Decode_Net1(n_channels=64, n_classes=1)
    net2 = Decode_Net2(n_channels=64, n_classes=1)

    # 将网络拷贝到deivce中
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)
    data_path = r"C:\Users\423\Desktop\铁测试\2.2m\水100x100\2.2m_water.npy"
    pred_net(net0, net1, net2, device, data_path)
