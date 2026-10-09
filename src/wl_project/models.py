from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout, stride=2):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.SiLU(),
    )


class Encoder(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.net = nn.Sequential(block(1,base), block(base,base*2), block(base*2,base*4), block(base*4,base*8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim=base*8
    def forward(self,x):
        return self.pool(self.net(x)).flatten(1)


class MultiTaskWeakLensingNet(nn.Module):
    """Shared encoder + heteroscedastic cosmology head + nuisance head."""
    def __init__(self, base_channels=24, dropout=0.1, multiscale=False):
        super().__init__(); self.multiscale=multiscale
        self.encoder=Encoder(base_channels)
        d=self.encoder.out_dim
        if multiscale:
            self.coarse_encoder=Encoder(max(8,base_channels//2)); d += self.coarse_encoder.out_dim
        self.shared=nn.Sequential(nn.Linear(d,256),nn.SiLU(),nn.Dropout(dropout))
        self.cosmo=nn.Linear(256,4)  # 2 means + 2 raw scales
        self.nuisance=nn.Linear(256,3)
    def forward(self,x):
        z=[self.encoder(x)]
        if self.multiscale:
            coarse=F.avg_pool2d(x, kernel_size=4, stride=4)
            z.append(self.coarse_encoder(coarse))
        h=self.shared(torch.cat(z,dim=1))
        c=self.cosmo(h)
        mean=c[:,:2]
        sigma=F.softplus(c[:,2:]) + 1e-4
        nuisance=self.nuisance(h)
        return mean,sigma,nuisance
