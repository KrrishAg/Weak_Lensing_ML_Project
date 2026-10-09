#!/usr/bin/env python
"""Train the controlled baseline/proposed multitask ensemble and evaluate inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wl_project.data import RepeatDataset, estimate_image_stats, make_dataset, make_splits
from wl_project.metrics import (
    coverage_at_levels,
    fit_sigma_scale,
    nuisance_metrics,
    phase1_score,
    regression_metrics,
)
from wl_project.models import MultiTaskWeakLensingNet
from wl_project.training import predict, train_model
from wl_project.utils import load_yaml, resolve_device, save_json, seed_everything


def physical_outputs(member_outputs, scaler):
    member_means = np.stack([x[0] for x in member_outputs])
    member_sigmas = np.stack([x[1] for x in member_outputs])
    member_nuis = np.stack([x[2] for x in member_outputs])
    y_scaled = member_outputs[0][3]

    mean_z = member_means.mean(axis=0)
    sigma_z = np.sqrt(np.mean(member_sigmas**2, axis=0) + np.var(member_means, axis=0))
    nuis_z = member_nuis.mean(axis=0)

    truth = scaler.inverse(y_scaled)
    mean = mean_z * scaler.std[:2] + scaler.mean[:2]
    sigma = sigma_z * scaler.std[:2]
    nuis = nuis_z * scaler.std[2:] + scaler.mean[2:]
    return member_means, member_sigmas, member_nuis, y_scaled, truth, mean, sigma, nuis


def evaluate_split(member_outputs, scaler, sigma_scale=None):
    _, _, _, _, truth, mean, sigma, nuis = physical_outputs(member_outputs, scaler)
    raw = regression_metrics(truth[:, :2], mean, sigma)
    raw.update(coverage_at_levels(truth[:, :2], mean, sigma))
    raw["phase1_score"] = phase1_score(truth[:, :2], mean, sigma)
    raw.update(nuisance_metrics(truth[:, 2:], nuis))
    if sigma_scale is None:
        return raw, None
    calibrated_sigma = sigma * np.asarray(sigma_scale)[None, :]
    calibrated = regression_metrics(truth[:, :2], mean, calibrated_sigma)
    calibrated.update(coverage_at_levels(truth[:, :2], mean, calibrated_sigma))
    calibrated["phase1_score"] = phase1_score(truth[:, :2], mean, calibrated_sigma)
    calibrated.update(nuisance_metrics(truth[:, 2:], nuis))
    return raw, calibrated


def save_prediction_pack(path, outputs, scaler, image_mean, image_std, sigma_scale, split_info):
    member_means, member_sigmas, member_nuis, y_scaled, *_ = physical_outputs(outputs, scaler)
    np.savez(
        path,
        member_means=member_means,
        member_sigmas=member_sigmas,
        member_nuisance=member_nuis,
        y_scaled=y_scaled,
        label_mean=scaler.mean,
        label_std=scaler.std,
        image_mean=np.float32(image_mean),
        image_std=np.float32(image_std),
        sigma_scale=np.asarray(sigma_scale, dtype=np.float32),
        ensemble_size=np.int32(len(outputs)),
        split_strategy=np.array(split_info["strategy"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--ensemble-size", type=int)
    ap.add_argument("--nuisance-weight", type=float)
    ap.add_argument("--tag", default="proposed")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    d = cfg["data"]
    t = cfg["training"]
    mcfg = cfg["model"]
    ccfg = cfg.get("calibration", {"enabled": True, "target_coverage": 0.68})
    if args.ensemble_size is not None:
        t["ensemble_size"] = args.ensemble_size
    if args.nuisance_weight is not None:
        t["nuisance_weight"] = args.nuisance_weight

    data_dir = ROOT / d["data_dir"]
    labels_path = data_dir / d["label_file"]
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {labels_path}. Download data first with scripts/download_data.py."
        )
    labels = np.load(labels_path, mmap_mode="r")
    train_idx, val_idx, test_idx, scaler, split_info = make_splits(
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
    save_json(ROOT / "results" / f"{args.tag}_split.json", split_info)
    print("split:", split_info)

    provisional = make_dataset(
        d, ROOT, train_idx, scaler, 0.0, 1.0, True, cfg["seed"], False
    )
    image_mean, image_std = estimate_image_stats(provisional, d["stats_samples"])
    print(f"global image statistics (train only): mean={image_mean:.6g}, std={image_std:.6g}")

    train_ds = make_dataset(
        d, ROOT, train_idx, scaler, image_mean, image_std, True, cfg["seed"], True
    )
    repeats = int(d.get("noise_repeats_per_epoch", 1))
    if repeats > 1:
        train_ds = RepeatDataset(train_ds, repeats)
        print(f"fresh noise realizations per map per epoch: {repeats}")
    val_ds = make_dataset(
        d, ROOT, val_idx, scaler, image_mean, image_std, True, cfg["seed"] + 1000, False
    )
    test_ds = make_dataset(
        d, ROOT, test_idx, scaler, image_mean, image_std, True, cfg["seed"] + 2000, False
    )

    device = resolve_device(t["device"])
    print("device:", device)
    print("dataset backend:", type(train_ds).__name__)

    val_outputs = []
    test_outputs = []
    for k in range(int(t["ensemble_size"])):
        member_seed = cfg["seed"] + k * 100
        seed_everything(member_seed)
        model = MultiTaskWeakLensingNet(
            mcfg["base_channels"], mcfg["dropout"], mcfg["multiscale"]
        )
        ckpt = ROOT / "checkpoints" / f"{args.tag}_member_{k}.pt"
        print(f"\n=== ensemble member {k + 1}/{t['ensemble_size']} seed={member_seed} ===")
        hist = train_model(
            model,
            train_ds,
            val_ds,
            device,
            epochs=t["epochs"],
            batch_size=t["batch_size"],
            lr=t["learning_rate"],
            weight_decay=t["weight_decay"],
            nuisance_weight=t["nuisance_weight"],
            num_workers=t["num_workers"],
            patience=t["patience"],
            grad_clip_norm=t.get("grad_clip_norm", 5.0),
            checkpoint=ckpt,
        )
        save_json(ROOT / "results" / f"{args.tag}_history_member_{k}.json", hist)
        val_outputs.append(
            predict(model, val_ds, device, t["batch_size"] * 2, num_workers=0)
        )
        test_outputs.append(
            predict(model, test_ds, device, t["batch_size"] * 2, num_workers=0)
        )

    # Calibration is fitted on validation only and evaluated on the untouched test split.
    _, _, _, _, val_truth, val_mean, val_sigma, _ = physical_outputs(val_outputs, scaler)
    if ccfg.get("enabled", True):
        sigma_scale = fit_sigma_scale(
            val_truth[:, :2],
            val_mean,
            val_sigma,
            target_coverage=ccfg.get("target_coverage", 0.68),
        )
    else:
        sigma_scale = np.ones(2, dtype=np.float32)
    print("sigma calibration factors [Omega_m, S8]:", sigma_scale)

    val_raw, val_cal = evaluate_split(val_outputs, scaler, sigma_scale)
    test_raw, test_cal = evaluate_split(test_outputs, scaler, sigma_scale)
    metrics = {
        "tag": args.tag,
        "nuisance_weight": float(t["nuisance_weight"]),
        "ensemble_size": int(t["ensemble_size"]),
        "image_mean": image_mean,
        "image_std": image_std,
        "sigma_scale": sigma_scale.tolist(),
        "validation_raw": val_raw,
        "validation_calibrated": val_cal,
        "test_raw": test_raw,
        "test_calibrated": test_cal,
    }
    save_json(ROOT / "results" / f"{args.tag}_metrics.json", metrics)
    print("\nFINAL TEST METRICS")
    print(metrics["test_calibrated"] if test_cal is not None else test_raw)

    save_prediction_pack(
        ROOT / "results" / f"{args.tag}_val_predictions.npz",
        val_outputs,
        scaler,
        image_mean,
        image_std,
        sigma_scale,
        split_info,
    )
    save_prediction_pack(
        ROOT / "results" / f"{args.tag}_test_predictions.npz",
        test_outputs,
        scaler,
        image_mean,
        image_std,
        sigma_scale,
        split_info,
    )


if __name__ == "__main__":
    main()
