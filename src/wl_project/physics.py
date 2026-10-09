from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


def radial_power_spectrum(image: np.ndarray, bins: int = 24) -> np.ndarray:
    x=np.asarray(image,dtype=np.float32)
    x=x-np.mean(x)
    p=np.abs(np.fft.rfft2(x))**2
    ky=np.fft.fftfreq(x.shape[0])[:,None]
    kx=np.fft.rfftfreq(x.shape[1])[None,:]
    kr=np.sqrt(kx*kx+ky*ky)
    edges=np.linspace(0,kr.max()+1e-12,bins+1)
    idx=np.clip(np.digitize(kr.ravel(),edges)-1,0,bins-1)
    sums=np.bincount(idx,weights=p.ravel(),minlength=bins)
    counts=np.bincount(idx,minlength=bins)
    return (sums/np.maximum(counts,1)).astype(np.float32)


class PowerSpectrumReference:
    """Physics consistency model: cosmology -> expected binned power spectrum."""
    def __init__(self,bins=24,alpha=1.0):
        self.bins=bins; self.alpha=alpha
        self.model=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=alpha))
        self.resid_scale=None
    def fit(self, maps: np.ndarray, cosmo: np.ndarray):
        feats=np.stack([np.log1p(radial_power_spectrum(m,self.bins)) for m in maps])
        self.model.fit(cosmo,feats)
        resid=feats-self.model.predict(cosmo)
        self.resid_scale=np.maximum(resid.std(0),1e-4)
        return self
    def score(self,maps: np.ndarray,pred_cosmo: np.ndarray):
        obs=np.stack([np.log1p(radial_power_spectrum(m,self.bins)) for m in maps])
        exp=self.model.predict(pred_cosmo)
        z=(obs-exp)/self.resid_scale
        return np.mean(z*z,axis=1)
