from model.model import Encode_Net, Decode_Net1, Decode_Net2
from dataset import HSI_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
import torch.nn.functional as F


def train_net(net0, net1, net2, device, npyfile, epochs, batch_size, lr):
    # Load dataset
    HSI_dataset = HSI_Loader(npyfile)
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset, batch_size=batch_size, shuffle=True)
    # 3 nets use the same RMSprop and learning rate
    optimizer0 = optim.RMSprop(net0.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    optimizer1 = optim.RMSprop(net1.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    optimizer2 = optim.RMSprop(net2.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    # Best_loss, start with 'inf'
    best_loss = float('inf')
    # Train
    for epoch in range(epochs):
        # Train mode
        net0.train()
        net1.train()
        net2.train()
        for curve, label in train_loader:
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # Load data and label to device, curve add 1 dim so that it can feed into the net
            curve = curve.reshape(len(curve), 1, -1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            encode_out = net0(curve)
            a = net1(encode_out)
            bb = net2(encode_out)
            # Bathymetric model-based reconstruction
            u= bb / (a + bb)
            r = (0.084 + 0.170 * u) * u
            r = torch.squeeze(r)
            # Here, r and label both [1024,176]
            # Define MSE Loss
            MSE_Loss = nn.MSELoss()
            # Define SA(Spectral Angle) Loss
            def SA_Loss(r, label):
                r_l2_norm = torch.norm(r, p=2, dim=1) # [1024]
                label_l2_norm = torch.norm(label, p=2, dim=1) # [1024]
                SALoss =  torch.sum(torch.acos(torch.sum(r*label, dim=1)/(r_l2_norm*label_l2_norm)))
                SALoss /= math.pi * len(r)
                return SALoss
            # First, train using only MSE and determine 1e7 according to the convergence magnitude
            loss = 1e7*MSE_Loss(r, label) + SA_Loss(r, label)
            # Save the model parameters with the lowest loss
            if loss < best_loss:
                best_loss = loss
                torch.save(net0.state_dict(), 'best_model_net0.pth')
                torch.save(net1.state_dict(), 'best_model_net1.pth')
                torch.save(net2.state_dict(), 'best_model_net2.pth')
            # Back propagation, update parameters
            loss.backward()
            optimizer0.step()
            optimizer1.step()
            optimizer2.step()
        print(f'epoch:{epoch}/{epochs}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


if __name__ == "__main__":

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load nets
    net0 = Encode_Net()
    net1 = Decode_Net1()
    net2 = Decode_Net2()
    # Load  nets to device
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)
    # Give dataset, start train!
    npyfile = r"xxx\xxx_water.npy"
    train_net(net0, net1, net2, device, npyfile, epochs=1000, batch_size=1024, lr=0.00001)