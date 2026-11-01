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


# Reference bounds (tlo, thi, flo, fhi) = training-label ranges of TAGN and f0.
PRIOR_BOUNDS=(7.2,8.5,0.0,0.0265)


def nuisance_prior_violation(nuisance_pred: np.ndarray, bounds=PRIOR_BOUNDS) -> np.ndarray:
    """TAGN and f0: squared distance outside the [lo,hi] windows; dz: Gaussian surprisal proxy."""
    tagn,f0,dz=nuisance_pred.T
    tlo,thi,flo,fhi=bounds
    dt=np.maximum(tlo-tagn,0)+np.maximum(tagn-thi,0)
    df=np.maximum(flo-f0,0)+np.maximum(f0-fhi,0)
    return (dt/(thi-tlo))**2 + (df/(fhi-flo))**2 + (dz/0.022)**2


def fit_standardizers(*ind_signals):
    return [ScoreStandardizer.fit(x) for x in ind_signals]


def fuse(signals, standardizers, weights=(1,1,1)):
    z=[s.transform(x) for s,x in zip(standardizers,signals)]
    w=np.asarray(weights,float); w=w/np.sum(np.abs(w))
    return sum(wi*zi for wi,zi in zip(w,z))


def weight_grid(step=0.05):
    """All non-negative 3-weight combinations on the simplex (w1+w2+w3=1)."""
    n=int(round(1/step))
    return np.array([(i/n,j/n,(n-i-j)/n) for i in range(n+1) for j in range(n+1-i)])


def tune_fusion_weights(ind_signals, ood_signal_sets, standardizers, step=0.05):
    """Grid-search fusion weights on validation data.

    Objective: mean low-FPR TPR over all validation shifts (tie-break: mean AUROC).
    ood_signal_sets maps shift name -> signal triplet computed on shifted validation maps.
    """
    from .metrics import ood_metrics
    rows=[]
    for w in weight_grid(step):
        ind=fuse(ind_signals,standardizers,w)
        m=[ood_metrics(ind,fuse(s,standardizers,w)) for s in ood_signal_sets.values()]
        rows.append({"weights":[float(x) for x in w],
                     "val_low_fpr_tpr":float(np.mean([x["low_fpr_tpr"] for x in m])),
                     "val_auroc":float(np.mean([x["auroc"] for x in m]))})
    rows.sort(key=lambda r:(r["val_low_fpr_tpr"],r["val_auroc"]),reverse=True)
    return rows[0]["weights"],rows


def tune_prior_bounds(ind_nuis, ood_nuis_sets, grid):
    """Grid-search (tlo, thi, flo, fhi) for the nuisance-prior score on validation data.

    grid: dict with lists tagn_lo, tagn_hi, f0_lo, f0_hi (reference values listed first,
    so ties keep them). Objective: mean low-FPR TPR over validation shifts (tie-break AUROC).
    """
    from itertools import product
    from .metrics import ood_metrics
    rows=[]
    for b in product(grid["tagn_lo"],grid["tagn_hi"],grid["f0_lo"],grid["f0_hi"]):
        if b[0]>=b[1] or b[2]>=b[3]: continue
        ind=nuisance_prior_violation(ind_nuis,b)
        m=[ood_metrics(ind,nuisance_prior_violation(o,b)) for o in ood_nuis_sets.values()]
        rows.append({"bounds":[float(x) for x in b],
                     "val_low_fpr_tpr":float(np.mean([x["low_fpr_tpr"] for x in m])),
                     "val_auroc":float(np.mean([x["auroc"] for x in m]))})
    rows.sort(key=lambda r:(r["val_low_fpr_tpr"],r["val_auroc"]),reverse=True)
    return rows[0]["bounds"],rows


def synthetic_shift(image: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    rng=np.random.default_rng(seed); x=np.asarray(image,dtype=np.float32)
    if kind=="extra_noise": return x+rng.normal(0,max(x.std()*0.35,1e-4),x.shape)
    if kind=="smooth": return gaussian_filter(x,sigma=1.5)
    if kind=="amplitude": return x*1.30
    if kind=="highfreq":
        blur=gaussian_filter(x,sigma=1.2); return x+0.8*(x-blur)
    raise ValueError(f"Unknown shift: {kind}")
