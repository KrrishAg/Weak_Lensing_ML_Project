#!/usr/bin/env python
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np, torch
from torch.utils.data import random_split
from wl_project.data import SyntheticWeakLensingDataset
from wl_project.models import MultiTaskWeakLensingNet
from wl_project.training import train_model,predict
from wl_project.metrics import regression_metrics,ood_metrics
from wl_project.ood import ensemble_disagreement,nuisance_prior_violation,fit_standardizers,fuse


def main():
    torch.set_num_threads(2)
    ds=SyntheticWeakLensingDataset(n=96,height=48,width=32,seed=7)
    tr,va=random_split(ds,[72,24],generator=torch.Generator().manual_seed(1))
    members=[]; pred=[]
    for seed in [1,2]:
        torch.manual_seed(seed); m=MultiTaskWeakLensingNet(base_channels=8)
        train_model(m,tr,va,torch.device('cpu'),epochs=2,batch_size=12,lr=2e-3,patience=2)
        p=predict(m,va,torch.device('cpu'),batch_size=24); members.append(m); pred.append(tuple(p))
    member_means=np.stack([p[0] for p in pred])
    mean=member_means.mean(0); sigma=np.sqrt(np.mean(np.stack([p[1]**2 for p in pred]),0)+np.var(member_means,0))
    y=pred[0][3]
    print('synthetic regression metrics:',regression_metrics(y[:,:2],mean,sigma))
    dis=ensemble_disagreement(member_means); nuis=nuisance_prior_violation(pred[0][2])
    # smoke-test fusion using a third independent proxy signal
    ps=np.mean(np.abs(mean),axis=1)
    stds=fit_standardizers(dis,nuis,ps); ind=fuse([dis,nuis,ps],stds)
    ood=ind+np.linspace(0.5,2.0,len(ind))
    print('synthetic OoD metrics:',ood_metrics(ind,ood))
    print('SMOKE TEST PASSED')

if __name__=='__main__': main()
