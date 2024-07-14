import itertools
import torch
from torch import nn


class RelativeQuaternion(nn.Module):
    def __init__(self, parents):
        super().__init__()

        P = torch.tensor(parents, dtype=torch.float32, requires_grad=False)
        self.register_buffer('P', P)

    def forward(self, input):
        p = torch.einsum('vw,ncvt->ncwt', (self.P.data, input))
        p[:, 0:3] *= -1
        p_inv = p / ((p * p).sum(dim=1, keepdim=True) + 1e-4)

        a = p_inv[:, 3]
        b = p_inv[:, 0]
        c = p_inv[:, 1]
        d = p_inv[:, 2]

        t = input[:, 3]
        x = input[:, 0]
        y = input[:, 1]
        z = input[:, 2]

        na = a * t - b * x - c * y - d * z
        nb = b * t + a * x + c * z - d * y
        nc = c * t + a * y + b * z - d * x
        nd = d * t + z * a + c * x - b * y

        return torch.stack((nb, nc, nd, na), dim=1)
