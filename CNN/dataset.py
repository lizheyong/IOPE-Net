from torch.utils.data import Dataset
import numpy as np
import torch


class HSI_Loader(Dataset):

    def __init__(self, npyfile):
        # load dataset from 'npyfile'
        self.all_curve = np.load(npyfile)

    def __getitem__(self, index):
        curve = torch.tensor(self.all_curve[index])
        # label is curve itself
        label = torch.tensor(self.all_curve[index])
        return curve, label

    def __len__(self):
        return len(self.all_curve)


if __name__ == "__main__":

    # load dataset, 'water_curves.npy' has been flatten with the shape of [pixels, wavelength]
    HSI_dataset = HSI_Loader('data/water_curves.npy')
    print("number of loaded pixels：", len(HSI_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset, batch_size=1024, shuffle=True)
    for curve, label in train_loader:
        # get the length of this batch, especially for the last batch, may less than 'batch_size'
        len_of_this_batch = curve.shape[0]
        # before input the Net, it should add one dim(channel) after dim(batch)
        print(curve.reshape(len_of_this_batch, 1, -1).shape)
        break