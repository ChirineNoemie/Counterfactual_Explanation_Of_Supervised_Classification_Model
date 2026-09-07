import numpy as np
import torch
from torch import nn


class RBFKernel(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    # def forward(self, x1, x2):
    #     '''

    #     :param x1: x (N x D)
    #     :param x2: xis (SVs x D)
    #     :return: (N x SVs)
    #     '''
    #     dist = torch.sum((x1[:, None] - x2) ** 2, dim=2)
    #     return torch.exp(-self.gamma * dist)
    def forward(self, x1, x2):
        if isinstance(x1, np.ndarray):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if isinstance(x2, np.ndarray):
            x2 = torch.tensor(x2, dtype=torch.float32)
        dist = torch.sum((x1[:, None] - x2) ** 2, dim=2)
        return torch.exp(-self.gamma * dist)


class RBFKernelNumpy:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, xis, x):
        return np.exp(-self.gamma * np.sum((xis - x) ** 2, axis=1))


class RBFSVMModel:
    def __init__(self, alphas, xis, yis, b, gamma):
        self.alphas = alphas
        self.xis = xis
        self.yis = yis
        self.b = b
        self.gamma = gamma
        self.kernel = RBFKernelNumpy(gamma)

    def decision_function(self, x):
        if isinstance(x, np.ndarray):
             x = torch.tensor(x, dtype=torch.float32)
        return np.sum(self.alphas * self.kernel(self.xis, x)) + self.b
    


class DifferentiableRBFSVMModel(nn.Module):
    # def __init__(self, alphas, support_vectors, yis, intercept, gamma):
    #     super(DifferentiableRBFSVMModel, self).__init__()
    #     self.alphas = alphas
    #     self.xis = support_vectors
    #     self.yis = yis
    #     self.intercept = intercept
    #     self.gamma = gamma
    #     self.kernel = RBFKernel(gamma)

    def __init__(self, alphas, support_vectors, yis, intercept, gamma):
        super(DifferentiableRBFSVMModel, self).__init__()
        self.alphas = torch.as_tensor(alphas, dtype=torch.float32)
        self.xis = torch.as_tensor(support_vectors, dtype=torch.float32)
        self.yis = torch.as_tensor(yis, dtype=torch.float32)
        self.intercept = torch.as_tensor(intercept, dtype=torch.float32)
        self.gamma = gamma
        self.kernel = RBFKernel(gamma)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        res = torch.sum(self.alphas * self.yis * self.kernel(x, self.xis), dim=1) + self.intercept
        res = torch.sigmoid(res)
        return res.unsqueeze(0)
