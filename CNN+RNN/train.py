from model.model import Encode_Net, Decode_Net1, Decode_Net2
from dataset import HSI_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
from log import Logger
import torch.nn.functional as F

sys.stdout = Logger()

def train_net(net0, net1, net2, device, npyfile, epochs=2000, batch_size=1024, lr=0.00001):
    # 加载训练集
    HSI_dataset = HSI_Loader(npyfile)
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer0 = optim.RMSprop(net0.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    optimizer1 = optim.RMSprop(net1.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    optimizer2 = optim.RMSprop(net2.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    # 定义loss
    criterion = nn.MSELoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net0.train()
        net1.train()
        net2.train()
        # 按照batch_size开始训练
        for curve, label in train_loader:
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # 将数据拷贝到device中
            curve = curve.reshape(len(curve), 1, -1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            encode_out = net0(curve)
            a = net1(encode_out)
            b = net2(encode_out)
            u= b / (a + b)
            r = (0.084 + 0.170 * u) * u
            r = torch.squeeze(r)
            """
            r, label都是[1024,176]
            """
            # 定义sa光谱角损失函数
            def SALoss(r, label):
                # 下面得到每行的二范数，也是再用哈达玛积相乘
                r_l2_norm = torch.norm(r, p=2, dim=1) # [1024]
                label_l2_norm = torch.norm(label, p=2, dim=1) # [1024]
                # r*label,对应元素乘,hadamard积,[1024,176],然后，每行求和torch.sum(r*label,dim=1)
                # 这样得到的是“向量r与向量label的内积”
                SALoss =  torch.sum(
                    torch.acos(torch.sum(r*label, dim=1)/(r_l2_norm*label_l2_norm))
                ) # acos括号内为[1024]
                SALoss /= math.pi * len(r) # 除以pi归一化到[0,1]，除以batch_size平均一下
                return SALoss
            # 计算loss
            # mseloss的量级为4e-8,所以乘e7,但是会不会一开始太大
            loss = 1e7*criterion(r, label) + SALoss(r, label)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net0.state_dict(), 'best_model_net0.pth')
                torch.save(net1.state_dict(), 'best_model_net1.pth')
                torch.save(net2.state_dict(), 'best_model_net2.pth')
            # 更新参数
            loss.backward()
            optimizer0.step()
            optimizer1.step()
            optimizer2.step()
        print(f'epoch:{epoch}, loss:{loss.item()}')
    print(best_loss)

if __name__ == "__main__":

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络，曲线单通道1，分类为2。
    net0 = Encode_Net(n_channels=1, n_classes=64)
    net1 = Decode_Net1(n_channels=64, n_classes=1)
    net2 = Decode_Net2(n_channels=64, n_classes=1)

    # 将网络拷贝到deivce中
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)

    # 指定训练集地址，开始训练
    data_path = 'data/all_curve.npy'
    train_net(net0, net1, net2, device, data_path)