import torch
from torch.utils.data import Dataset


def logSumExp(x, axis=None):
    xMax = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - xMax)).sum(axis)) + xMax
    return y


class RegressionDataset(Dataset):
    def __init__(self, xTensor, yTensor):
        self.x = xTensor
        self.y = yTensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
