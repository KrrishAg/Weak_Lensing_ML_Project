from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .ood import synthetic_shift


class ArrayFieldDataset(Dataset):
    def __init__(self, maps: np.ndarray, labels_scaled: np.ndarray | None = None):
        self.maps=np.asarray(maps,dtype=np.float32)
        self.labels=None if labels_scaled is None else np.asarray(labels_scaled,dtype=np.float32)
    def __len__(self): return len(self.maps)
    def __getitem__(self,i):
        x=torch.from_numpy(self.maps[i][None])
        if self.labels is None: return x
        return x,torch.from_numpy(self.labels[i])


def materialize_maps(dataset, indices: np.ndarray | list[int]) -> tuple[np.ndarray,np.ndarray]:
    xs=[]; ys=[]
    for i in indices:
        x,y=dataset[int(i)]; xs.append(x.numpy()[0]); ys.append(y.numpy())
    return np.stack(xs),np.stack(ys)


def shifted_maps(maps: np.ndarray, kind: str, seed: int=0) -> np.ndarray:
    return np.stack([synthetic_shift(m,kind,seed+i) for i,m in enumerate(maps)]).astype(np.float32)


def predict_array(model, maps: np.ndarray, device, batch_size=64):
    loader=DataLoader(ArrayFieldDataset(maps),batch_size=batch_size,shuffle=False)
    means=[]; sigmas=[]; nuisance=[]
    model.eval()
    with torch.no_grad():
        for x in loader:
            m,s,n=model(x.to(device)); means.append(m.cpu().numpy()); sigmas.append(s.cpu().numpy()); nuisance.append(n.cpu().numpy())
    return np.concatenate(means),np.concatenate(sigmas),np.concatenate(nuisance)
