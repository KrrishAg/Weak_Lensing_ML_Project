from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class ScoreStandardizer:
    mean: float
    std: float
    @classmethod
    def fit(cls,x):
        x=np.asarray(x,float); return cls(float(x.mean()),float(max(x.std(),1e-8)))
    def transform(self,x): return (np.asarray(x)-self.mean)/self.std


def ensemble_disagreement(member_means: np.ndarray) -> np.ndarray:
    """member_means shape [M,N,2]."""
    return np.mean(np.var(member_means,axis=0),axis=1)


def nuisance_prior_violation(nuisance_pred: np.ndarray) -> np.ndarray:
    """TAGN and f0: squared distance outside uniform prior; dz: Gaussian surprisal proxy."""
    tagn,f0,dz=nuisance_pred.T
    tlo,thi=7.2,8.5; flo,fhi=0.0,0.0265
    dt=np.maximum(tlo-tagn,0)+np.maximum(tagn-thi,0)
    df=np.maximum(flo-f0,0)+np.maximum(f0-fhi,0)
    return (dt/(thi-tlo))**2 + (df/(fhi-flo))**2 + (dz/0.022)**2


def fit_standardizers(*ind_signals):
    return [ScoreStandardizer.fit(x) for x in ind_signals]


def fuse(signals, standardizers, weights=(1,1,1)):
    z=[s.transform(x) for s,x in zip(standardizers,signals)]
    w=np.asarray(weights,float); w=w/np.sum(np.abs(w))
    return sum(wi*zi for wi,zi in zip(w,z))


def synthetic_shift(image: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    rng=np.random.default_rng(seed); x=np.asarray(image,dtype=np.float32)
    if kind=="extra_noise": return x+rng.normal(0,max(x.std()*0.35,1e-4),x.shape)
    if kind=="smooth": return gaussian_filter(x,sigma=1.5)
    if kind=="amplitude": return x*1.30
    if kind=="highfreq":
        blur=gaussian_filter(x,sigma=1.2); return x+0.8*(x-blur)
    raise ValueError(f"Unknown shift: {kind}")
