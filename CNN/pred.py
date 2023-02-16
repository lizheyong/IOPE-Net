from model.model import Encode_Net, Decode_Net1, Decode_Net2
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from dataset import HSI_Loader
import torch.nn as nn
import numpy as np
import torch
import math
torch.set_printoptions(precision=16, threshold=float('inf'))


def pred_net(net0, net1, net2, device, npyfile, batch_size):
    # Load dataset
    HSI_dataset = HSI_Loader(npyfile)
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset, batch_size=batch_size, shuffle=False)
    # Load the trained network model parameters
    net0.load_state_dict(torch.load('best_model_net0.pth', map_location=device))
    net1.load_state_dict(torch.load('best_model_net1.pth', map_location=device))
    net2.load_state_dict(torch.load('best_model_net2.pth', map_location=device))
    # Pred mode
    net0.eval()
    net1.eval()
    net2.eval()
    for curve, label in train_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        encode_out = net0(curve)
        a = net1(encode_out)
        bb = net2(encode_out)
        u = bb / (a+bb)
        r = (0.084 + 0.170 * u) * u
        r = torch.squeeze(r)
        # select a curve to plot
        pltcurve = 100
        print(f'r:{r[pltcurve,:]}')
        print(f'a:{a[pltcurve,0,:]}')
        print(f'bb:{bb[pltcurve,0,:]}')
        print(f'label:{label[pltcurve,:]}')
        MSE_Loss = nn.MSELoss()
        def SA_Loss(r, label):
            r_l2_norm = torch.norm(r, p=2, dim=1)  # [1024]
            label_l2_norm = torch.norm(label, p=2, dim=1)  # [1024]
            SALoss = torch.sum(torch.acos(torch.sum(r * label, dim=1) / (r_l2_norm * label_l2_norm)))
            SALoss /= math.pi * len(r)
            return SALoss
        loss = 1e7 * MSE_Loss(r, label) + SA_Loss(r, label)
        # REP(relative error percentage)
        REP = torch.abs(r-label)/label # REP [1024,176]
        sum_REP = torch.where(torch.isinf(torch.sum(REP, 1)), torch.full_like(torch.sum(REP, 1), 1), torch.sum(REP, 1)) # some 'inf' change to '1'
        mean_REP = torch.mean(sum_REP)
        print(f'Loss:{loss.item()}')
        print(f'REP:{torch.sum(REP[pltcurve,:]).item()}')
        print(f'mean_REP:{mean_REP.item()}')

        # Plot real(label) and reconstruct curve
        wavelength = np.load(r"xxxx\wavelength.npy")
        plt.figure()
        # Gaussian filtering can be removed
        plt.plot(wavelength, gaussian_filter1d(torch.squeeze(r[pltcurve,:]).detach().cpu().numpy(),sigma=5),
                 label='Reconstruct', color='r', marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(torch.squeeze(label[pltcurve,:]).detach().cpu().numpy(),sigma=5),
                 label='Real', color='b', marker='o', markersize=3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance Value')
        plt.legend()
        # Plot estimated 'a', 'bb'
        plt.figure()
        a_smoothed = gaussian_filter1d(torch.squeeze(a[pltcurve,:]).detach().cpu().numpy(), sigma=5)
        plt.plot(wavelength, a_smoothed,
                 label='a', color='r', marker='o', markersize=3)
        b_smoothed = gaussian_filter1d(torch.squeeze(bb[pltcurve,:]).detach().cpu().numpy(), sigma=5)
        plt.plot(wavelength, b_smoothed,
                 label='bb', color='b', marker='o', markersize=3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('m^(-1)')
        plt.legend()
        plt.show()
        break


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net0 = Encode_Net()
    net1 = Decode_Net1()
    net2 = Decode_Net2(1)
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)
    # The training data can be used here
    data_path = r"xxx\xxx_water.npy"
    pred_net(net0, net1, net2, device, npyfile, batch_size=1024)
