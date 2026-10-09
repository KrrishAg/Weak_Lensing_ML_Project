from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import multitask_loss


def seed_worker(worker_id: int) -> None:
    """Make NumPy/Python augmentation RNGs reproducible inside DataLoader workers."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(dataset, batch_size, shuffle, num_workers, device):
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def train_model(
    model,
    train_ds,
    val_ds,
    device,
    epochs=20,
    batch_size=32,
    lr=5e-4,
    weight_decay=1e-4,
    nuisance_weight=0.35,
    num_workers=0,
    patience=5,
    grad_clip_norm=5.0,
    checkpoint=None,
):
    train_loader = _loader(train_ds, batch_size, True, num_workers, device)
    val_loader = _loader(val_ds, batch_size, False, num_workers, device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )
    best = float("inf")
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, _ = multitask_loss(model(x), y, nuisance_weight)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()
            tr += float(loss.detach()) * len(x)
            n += len(x)

        model.eval()
        va = 0.0
        nv = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss, _ = multitask_loss(model(x), y, nuisance_weight)
                va += float(loss) * len(x)
                nv += len(x)

        tr /= max(n, 1)
        va /= max(nv, 1)
        sched.step(va)
        lr_now = float(opt.param_groups[0]["lr"])
        history.append(
            {"epoch": epoch, "train_loss": tr, "val_loss": va, "learning_rate": lr_now}
        )
        print(f"epoch={epoch:02d} train={tr:.5f} val={va:.5f} lr={lr_now:.2e}")

        if va < best:
            best = va
            bad = 0
            if checkpoint:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping after {epoch} epochs")
                break

    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
    return history


def predict(model, dataset, device, batch_size=64, num_workers=0):
    loader = _loader(dataset, batch_size, False, num_workers, device)
    model.eval()
    means, sigmas, nuis, ys = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            m, s, n = model(x.to(device, non_blocking=True))
            means.append(m.cpu().numpy())
            sigmas.append(s.cpu().numpy())
            nuis.append(n.cpu().numpy())
            ys.append(y.numpy())
    return (
        np.concatenate(means),
        np.concatenate(sigmas),
        np.concatenate(nuis),
        np.concatenate(ys),
    )
