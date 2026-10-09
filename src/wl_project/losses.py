import torch


def gaussian_nll(mean, sigma, truth):
    residual2=(mean-truth)**2
    return torch.mean(torch.sum(residual2/(sigma**2) + torch.log(sigma**2), dim=1))


def multitask_loss(outputs, labels, nuisance_weight=0.35):
    mean,sigma,nuisance=outputs
    lc=gaussian_nll(mean,sigma,labels[:,:2])
    ln=torch.mean((nuisance-labels[:,2:])**2)
    return lc+nuisance_weight*ln, {"cosmo_nll":float(lc.detach()),"nuisance_mse":float(ln.detach())}
