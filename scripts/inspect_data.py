#!/usr/bin/env python
"""Validate shapes/ranges and create the first EDA figures from the official data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from wl_project.data import LABEL_NAMES, add_shape_noise, make_splits
from wl_project.utils import load_yaml, save_json


def reconstruct(compact_row: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.zeros(mask.shape, dtype=np.float32)
    x[mask] = np.asarray(compact_row, dtype=np.float32)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    d = cfg["data"]
    data_dir = ROOT / d["data_dir"]
    kappa_path = data_dir / d["kappa_file"]
    label_path = data_dir / d["label_file"]
    mask_path = data_dir / d["mask_file"]
    for p in [kappa_path, label_path, mask_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run scripts/download_data.py first.")

    kappa = np.load(kappa_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r").astype(bool)

    if kappa.shape[:2] != labels.shape[:2]:
        raise ValueError(f"Map and label axes disagree: {kappa.shape} vs {labels.shape}")
    if labels.shape[-1] != 5:
        raise ValueError(f"Expected 5 labels, got {labels.shape}")
    if mask.shape != (d["height"], d["width"]):
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    if kappa.shape[-1] != int(mask.sum()):
        raise ValueError(f"Observed-pixel axis {kappa.shape[-1]} != mask sum {int(mask.sum())}")

    train, val, test, scaler, split_info = make_splits(
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

    flat = np.asarray(labels).reshape(-1, 5)
    summary = {
        "kappa_shape": list(kappa.shape),
        "kappa_dtype": str(kappa.dtype),
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "mask_shape": list(mask.shape),
        "observed_pixels": int(mask.sum()),
        "observed_fraction": float(mask.mean()),
        "labels": {
            name: {
                "min": float(flat[:, j].min()),
                "max": float(flat[:, j].max()),
                "mean": float(flat[:, j].mean()),
                "std": float(flat[:, j].std()),
            }
            for j, name in enumerate(LABEL_NAMES)
        },
        "split": split_info,
        "train_label_mean": scaler.mean.tolist(),
        "train_label_std": scaler.std.tolist(),
    }
    save_json(ROOT / "results" / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.imshow(mask.T, origin="lower", aspect="auto")
    ax.set_title("HSC WIDE12H survey mask")
    ax.set_xlabel("map y")
    ax.set_ylabel("map x")
    fig.tight_layout()
    fig.savefig(fig_dir / "dataset_mask.png", dpi=180)
    plt.close(fig)

    c, s = train[0]
    x = reconstruct(kappa[c, s], mask)
    noisy = add_shape_noise(
        x,
        mask,
        ng=d["ng"],
        pixel_size_arcmin=d["pixel_size_arcmin"],
        sigma_epsilon=d.get("sigma_epsilon", 0.4),
        rng=np.random.default_rng(cfg["seed"]),
    )
    for arr, name, title in [
        (x, "dataset_example_noiseless.png", "Example noiseless convergence map"),
        (noisy, "dataset_example_noisy.png", "Same map with shape noise"),
    ]:
        fig, ax = plt.subplots(figsize=(4, 8))
        im = ax.imshow(arr.T, origin="lower", aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(flat[:, 0], flat[:, 1], s=4, alpha=0.25)
    axes[0].set_xlabel(r"$\Omega_m$")
    axes[0].set_ylabel(r"$S_8$")
    axes[0].set_title("Cosmological grid")
    axes[1].scatter(flat[:, 2], flat[:, 3], s=4, alpha=0.25)
    axes[1].set_xlabel(r"$T_{AGN}$")
    axes[1].set_ylabel(r"$f_0$")
    axes[1].set_title("Baryonic nuisance parameters")
    fig.tight_layout()
    fig.savefig(fig_dir / "dataset_label_space.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(flat[:, 4], bins=40)
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel("count")
    ax.set_title("Photometric-redshift nuisance distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "dataset_delta_z.png", dpi=180)
    plt.close(fig)

    print(f"\nWrote dataset summary to results/dataset_summary.json and EDA figures to {fig_dir}")


if __name__ == "__main__":
    main()
