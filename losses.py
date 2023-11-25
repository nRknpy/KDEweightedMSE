import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde


class KDEWeightedMSESl(nn.Module):
    def __init__(self, dataset, band_width, device, mode='divide', standardize=False, eps=1e-6):
        super(KDEWeightedMSESl, self).__init__()
        assert mode in ('divide', 'onemin')
        self.band_width = band_width
        self.mode = mode
        self.standardize = standardize
        self.eps = eps
        self.device = device
        self.kernel = self._kernel(dataset)

    def _kernel(self, dataset):
        dataloader = DataLoader(
            dataset=dataset, batch_size=128, shuffle=True, drop_last=False)
        x = []
        for data in dataloader:
            _, batch = data
            batch = batch.view(batch.shape[0], -1)
            x.append(batch.cpu().numpy())
        x = np.concatenate(x)
        print(x.shape)

        kernel = KernelDensity(
            kernel='gaussian', bandwidth=self.band_width).fit(x)
        return kernel

    def forward(self, pred, target):
        f = torch.exp(torch.tensor(self.kernel.score_samples(target.cpu())))
        if self.mode == 'divide':
            loss = torch.mean(torch.dot(torch.tensor(1 / (f + self.eps)).float().to(self.device),
                                        ((pred - target) ** 2).sum(1)))
        else:
            loss = torch.mean(torch.dot(1 - f, ((pred - target) ** 2).sum(1)))
        return loss


class KDEWeightedMSESc(nn.Module):
    def __init__(self, data, band_width, device, mode='divide', standardize=False, eps=1e-6):
        super(KDEWeightedMSESc, self).__init__()
        assert mode in ('divide', 'onemin')
        self.mode = mode
        self.band_width = band_width
        self.eps = eps
        self.device = device
        self.kernel = self._kernel(data)
        self.standardize = standardize
        if self.standardize:
            self.f_max = torch.max(torch.tensor(self.kernel(np.array(data).T)))

    def forward(self, pred, target):
        f = torch.tensor(self.kernel(target.cpu().T))
        if self.standardize:
            f = f / self.f_max
        if self.mode == 'divide':
            loss = torch.mean(torch.dot(
                1 / (f + self.eps).float().to(self.device), ((pred - target) ** 2).sum(1)))
        else:
            loss = torch.mean(torch.dot(1 - f, ((pred - target) ** 2).sum(1)))
        return loss

    def _kernel(self, data):
        data = np.array(data).T
        return gaussian_kde(data, bw_method=self.band_width)
