import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import KernelDensity


class KDEWeightedMSE(nn.Module):
    def __init__(self, dataset, band_width, mode, standardize=False, eps=1e-6):
        assert mode in ('divide', 'onemin')
        self.band_width = band_width
        self.mode = mode
        self.standardize = standardize
        self.eps = eps
        self.device = dataset.device

        self.kernel = self._kernel(dataset)

    def _kernel(self, dataset):
        dataloader = DataLoader(
            dataset=dataset, batch_size=128, shuffle=True, drop_last=False)
        x = []
        for data in dataloader:
            batch, _ = data
            batch = batch.view(batch.size[0], -1)
            x.append(batch.cpu().numpy())
        x = np.concatenate(x)

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
