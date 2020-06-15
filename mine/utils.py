import torch


def logSumExp(x, axis=None):
    xMax = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - xMax)).sum(axis)) + xMax
    return y
