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
    PRIOR_BOUNDS,
    ensemble_disagreement,
    fit_standardizers,
    fuse,
    nuisance_prior_violation,
    tune_fusion_weights,
    tune_prior_bounds,
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


def raw_signals(members, maps, device, batch, label_mean, label_std, psref):
    """Disagreement, predicted physical nuisances and power-spectrum score for a set of maps."""
    means, _, nuis = predict_ensemble(members, maps, device, batch)
    disagreement = ensemble_disagreement(means)
    nuisance_mean = physical_nuis(nuis.mean(axis=0), label_mean, label_std)
    cosmo = physical_mean(means.mean(axis=0), label_mean, label_std)
    ps = psref.score(maps, cosmo)
    return disagreement, nuisance_mean, ps


def with_prior(raw, bounds):
    """Signal triplet with the nuisance-prior score computed for the given bounds."""
    return raw[0], nuisance_prior_violation(raw[1], bounds), raw[2]


def mean_val_metrics(ind, ood_sets):
    m = [ood_metrics(ind, o) for o in ood_sets]
    return {
        "val_low_fpr_tpr": float(np.mean([x["low_fpr_tpr"] for x in m])),
        "val_auroc": float(np.mean([x["auroc"] for x in m])),
    }


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
    batch = t["batch_size"] * 2
    val_maps, _ = choose(val_ds, min(n, len(val_ds)), rng)
    val_raw = raw_signals(members, val_maps, device, batch, label_mean, label_std, psref)
    ref_bounds = [float(b) for b in ocfg.get("prior_bounds", PRIOR_BOUNDS)]
    val_signals = with_prior(val_raw, ref_bounds)
    standardizers = fit_standardizers(*val_signals)

    shifts = ["extra_noise", "smooth", "amplitude", "highfreq"]
    tune_bounds = ocfg.get("tune_prior_bounds", False)
    tune_weights = ocfg.get("tune_fusion_weights", False)

    # Hyperparameter tuning (prior bounds, then fusion weights) uses validation maps only,
    # shifted with seeds disjoint from the test shifts. Test maps remain untouched.
    if tune_bounds or tune_weights:
        val_ood_raw = {}
        for j, kind in enumerate(shifts):
            vmaps = shifted_maps(val_maps, kind, cfg["seed"] + 500000 + j * 10000)
            val_ood_raw[kind] = raw_signals(
                members, vmaps, device, batch, label_mean, label_std, psref
            )

    tuned_bounds = ref_bounds
    if tune_bounds:
        val_ood_nuis = {k: r[1] for k, r in val_ood_raw.items()}
        tuned_bounds, bgrid = tune_prior_bounds(val_raw[1], val_ood_nuis, ocfg["prior_grid"])
        ref_val = mean_val_metrics(
            nuisance_prior_violation(val_raw[1], ref_bounds),
            [nuisance_prior_violation(o, ref_bounds) for o in val_ood_nuis.values()],
        )
        save_json(
            ROOT / "results" / f"{args.tag}_prior_tuning.json",
            {
                "objective": "nuisance-prior score alone: mean low-FPR TPR over validation shifts (tie-break mean AUROC)",
                "parameters": ["tagn_lo", "tagn_hi", "f0_lo", "f0_hi"],
                "grid": ocfg["prior_grid"],
                "n_candidates": len(bgrid),
                "best_bounds": tuned_bounds,
                "best": bgrid[0],
                "reference_bounds": {"bounds": ref_bounds, **ref_val},
                "top10": bgrid[:10],
            },
        )
        print("tuned prior bounds [tagn_lo, tagn_hi, f0_lo, f0_hi]:", tuned_bounds)

    val_tuned = with_prior(val_raw, tuned_bounds)
    tuned_standardizers = fit_standardizers(*val_tuned)
    tuned_weights = [float(w) for w in ocfg["fusion_weights"]]
    if tune_weights:
        val_ood = {k: with_prior(r, tuned_bounds) for k, r in val_ood_raw.items()}
        tuned_weights, grid = tune_fusion_weights(
            val_tuned, val_ood, tuned_standardizers, ocfg.get("fusion_grid_step", 0.05)
        )
        fixed_val = mean_val_metrics(
            fuse(val_tuned, tuned_standardizers, ocfg["fusion_weights"]),
            [fuse(s, tuned_standardizers, ocfg["fusion_weights"]) for s in val_ood.values()],
        )
        save_json(
            ROOT / "results" / f"{args.tag}_fusion_tuning.json",
            {
                "objective": "mean low-FPR TPR over validation shifts (tie-break mean AUROC)",
                "prior_bounds_used": tuned_bounds,
                "grid_step": ocfg.get("fusion_grid_step", 0.05),
                "n_candidates": len(grid),
                "best_weights": tuned_weights,
                "best": grid[0],
                "fixed_weights": {"weights": [float(w) for w in ocfg["fusion_weights"]], **fixed_val},
                "top10": grid[:10],
            },
        )
        print("tuned fusion weights [disagreement, prior, power_spectrum]:", tuned_weights)

    test_maps, _ = choose(test_ds, min(n, len(test_ds)), rng)
    test_raw = raw_signals(members, test_maps, device, batch, label_mean, label_std, psref)
    ind_signals = with_prior(test_raw, ref_bounds)
    ind_prior_tuned = nuisance_prior_violation(test_raw[1], tuned_bounds)
    ind_hybrid = fuse(ind_signals, standardizers, ocfg["fusion_weights"])
    ind_tuned = fuse(with_prior(test_raw, tuned_bounds), tuned_standardizers, tuned_weights)

    report = {}
    arrays = {
        "ind_disagreement": ind_signals[0],
        "ind_prior": ind_signals[1],
        "ind_prior_tuned": ind_prior_tuned,
        "ind_power_spectrum": ind_signals[2],
        "ind_hybrid": ind_hybrid,
        "ind_hybrid_tuned": ind_tuned,
    }
    for j, kind in enumerate(shifts):
        omaps = shifted_maps(test_maps, kind, cfg["seed"] + j * 10000)
        oraw = raw_signals(members, omaps, device, batch, label_mean, label_std, psref)
        osig = with_prior(oraw, ref_bounds)
        hybrid = fuse(osig, standardizers, ocfg["fusion_weights"])
        report[kind] = {
            "ensemble_disagreement": ood_metrics(ind_signals[0], osig[0]),
            "nuisance_prior": ood_metrics(ind_signals[1], osig[1]),
            "power_spectrum": ood_metrics(ind_signals[2], osig[2]),
            "hybrid": ood_metrics(ind_hybrid, hybrid),
        }
        if tune_bounds:
            prior_tuned = nuisance_prior_violation(oraw[1], tuned_bounds)
            report[kind]["nuisance_prior_tuned"] = ood_metrics(ind_prior_tuned, prior_tuned)
            arrays[f"{kind}_prior_tuned"] = prior_tuned
        if tune_bounds or tune_weights:
            tuned = fuse(with_prior(oraw, tuned_bounds), tuned_standardizers, tuned_weights)
            report[kind]["hybrid_tuned"] = ood_metrics(ind_tuned, tuned)
            arrays[f"{kind}_hybrid_tuned"] = tuned
        arrays[f"{kind}_disagreement"] = osig[0]
        arrays[f"{kind}_prior"] = osig[1]
        arrays[f"{kind}_power_spectrum"] = osig[2]
        arrays[f"{kind}_hybrid"] = hybrid
        print(kind, json.dumps(report[kind], indent=2))

    save_json(ROOT / "results" / f"{args.tag}_ood_metrics.json", report)
    np.savez(ROOT / "results" / f"{args.tag}_ood_scores.npz", **arrays)


if __name__ == "__main__":
    main()
