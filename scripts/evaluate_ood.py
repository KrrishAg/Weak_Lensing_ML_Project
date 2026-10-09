#!/usr/bin/env python
"""Offline OoD evaluation using validation-fitted scores and an untouched test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wl_project.data import make_dataset, make_splits
from wl_project.evaluation import materialize_maps, predict_array, shifted_maps
from wl_project.metrics import ood_metrics
from wl_project.models import MultiTaskWeakLensingNet
from wl_project.ood import (
    ensemble_disagreement,
    fit_standardizers,
    fuse,
    nuisance_prior_violation,
)
from wl_project.physics import PowerSpectrumReference
from wl_project.utils import load_yaml, resolve_device, save_json


def load_members(tag, cfg, device, ensemble_size):
    out = []
    mc = cfg["model"]
    for k in range(int(ensemble_size)):
        path = ROOT / "checkpoints" / f"{tag}_member_{k}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint {path}")
        m = MultiTaskWeakLensingNet(
            mc["base_channels"], mc["dropout"], mc["multiscale"]
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        out.append(m)
    return out


def predict_ensemble(members, maps, device, batch_size):
    pred = [predict_array(m, maps, device, batch_size) for m in members]
    means = np.stack([p[0] for p in pred])
    sigmas = np.stack([p[1] for p in pred])
    nuis = np.stack([p[2] for p in pred])
    return means, sigmas, nuis


def physical_mean(means_scaled, label_mean, label_std):
    return means_scaled * label_std[:2] + label_mean[:2]


def physical_nuis(nuis_scaled, label_mean, label_std):
    return nuis_scaled * label_std[2:] + label_mean[2:]


def signal_triplet(members, maps, device, batch, label_mean, label_std, psref):
    means, _, nuis = predict_ensemble(members, maps, device, batch)
    disagreement = ensemble_disagreement(means)
    nuisance_mean = physical_nuis(nuis.mean(axis=0), label_mean, label_std)
    prior = nuisance_prior_violation(nuisance_mean)
    cosmo = physical_mean(means.mean(axis=0), label_mean, label_std)
    ps = psref.score(maps, cosmo)
    return disagreement, prior, ps


def choose(dataset, n, rng):
    n = min(int(n), len(dataset))
    ids = rng.choice(len(dataset), size=n, replace=False)
    return materialize_maps(dataset, ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--tag", default="proposed")
    ap.add_argument("--n", type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    d, t, pcfg, ocfg = cfg["data"], cfg["training"], cfg["physics"], cfg["ood"]
    pred_path = ROOT / "results" / f"{args.tag}_val_predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Train the model first: {pred_path}")
    pack = np.load(pred_path)
    label_mean = pack["label_mean"]
    label_std = pack["label_std"]
    image_mean = float(pack["image_mean"])
    image_std = float(pack["image_std"])
    ensemble_size = int(pack["ensemble_size"])

    data_dir = ROOT / d["data_dir"]
    labels = np.load(data_dir / d["label_file"], mmap_mode="r")
    train_idx, val_idx, test_idx, scaler, _ = make_splits(
        labels,
        strategy=d["split_strategy"],
        val_cosmologies=d["val_cosmologies"],
        test_cosmologies=d["test_cosmologies"],
        val_fraction=d["val_fraction"],
        test_fraction=d["test_fraction"],
        seed=d["split_seed"],
        max_train_samples=d.get("max_train_samples"),
        max_val_samples=d.get("max_val_samples"),
        max_test_samples=d.get("max_test_samples"),
    )
    scaler.mean = label_mean.astype(np.float32)
    scaler.std = label_std.astype(np.float32)

    train_ds = make_dataset(
        d, ROOT, train_idx, scaler, image_mean, image_std, True, cfg["seed"], False
    )
    val_ds = make_dataset(
        d, ROOT, val_idx, scaler, image_mean, image_std, True, cfg["seed"] + 1000, False
    )
    test_ds = make_dataset(
        d, ROOT, test_idx, scaler, image_mean, image_std, True, cfg["seed"] + 2000, False
    )

    device = resolve_device(t["device"])
    members = load_members(args.tag, cfg, device, ensemble_size)
    n = args.n or ocfg["synthetic_samples"]
    rng = np.random.default_rng(cfg["seed"] + 44)

    # Physics reference is trained only on training maps.
    pn = min(int(pcfg["ps_fit_samples"]), len(train_ds))
    train_maps, train_y = choose(train_ds, pn, rng)
    train_phys = train_y * label_std + label_mean
    psref = PowerSpectrumReference(pcfg["ps_bins"], pcfg["ps_ridge_alpha"]).fit(
        train_maps, train_phys[:, :2]
    )

    # Validation maps define the InD score normalization. Test maps remain untouched.
    val_maps, _ = choose(val_ds, min(n, len(val_ds)), rng)
    val_signals = signal_triplet(
        members,
        val_maps,
        device,
        t["batch_size"] * 2,
        label_mean,
        label_std,
        psref,
    )
    standardizers = fit_standardizers(*val_signals)

    test_maps, _ = choose(test_ds, min(n, len(test_ds)), rng)
    ind_signals = signal_triplet(
        members,
        test_maps,
        device,
        t["batch_size"] * 2,
        label_mean,
        label_std,
        psref,
    )
    ind_hybrid = fuse(ind_signals, standardizers, ocfg["fusion_weights"])

    shifts = ["extra_noise", "smooth", "amplitude", "highfreq"]
    report = {}
    arrays = {
        "ind_disagreement": ind_signals[0],
        "ind_prior": ind_signals[1],
        "ind_power_spectrum": ind_signals[2],
        "ind_hybrid": ind_hybrid,
    }
    for j, kind in enumerate(shifts):
        omaps = shifted_maps(test_maps, kind, cfg["seed"] + j * 10000)
        osig = signal_triplet(
            members,
            omaps,
            device,
            t["batch_size"] * 2,
            label_mean,
            label_std,
            psref,
        )
        hybrid = fuse(osig, standardizers, ocfg["fusion_weights"])
        report[kind] = {
            "ensemble_disagreement": ood_metrics(ind_signals[0], osig[0]),
            "nuisance_prior": ood_metrics(ind_signals[1], osig[1]),
            "power_spectrum": ood_metrics(ind_signals[2], osig[2]),
            "hybrid": ood_metrics(ind_hybrid, hybrid),
        }
        arrays[f"{kind}_disagreement"] = osig[0]
        arrays[f"{kind}_prior"] = osig[1]
        arrays[f"{kind}_power_spectrum"] = osig[2]
        arrays[f"{kind}_hybrid"] = hybrid
        print(kind, json.dumps(report[kind], indent=2))

    save_json(ROOT / "results" / f"{args.tag}_ood_metrics.json", report)
    np.savez(ROOT / "results" / f"{args.tag}_ood_scores.npz", **arrays)


if __name__ == "__main__":
    main()
