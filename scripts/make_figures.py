#!/usr/bin/env python
"""Create report-ready inference, calibration, training and nuisance plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np


def load_pack(tag):
    p = ROOT / "results" / f"{tag}_test_predictions.npz"
    if not p.exists():
        raise FileNotFoundError(f"Run training first: {p}")
    return np.load(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="proposed")
    args = ap.parse_args()
    a = load_pack(args.tag)

    means = a["member_means"]
    sigmas = a["member_sigmas"]
    nuis = a["member_nuisance"]
    y = a["y_scaled"]
    lm = a["label_mean"]
    ls = a["label_std"]
    scale = a["sigma_scale"]

    mean_z = means.mean(axis=0)
    sigma_z = np.sqrt(np.mean(sigmas**2, axis=0) + np.var(means, axis=0))
    truth = y * ls + lm
    pred = mean_z * ls[:2] + lm[:2]
    sigma_raw = sigma_z * ls[:2]
    sigma_cal = sigma_raw * scale[None, :]
    nuis_pred = nuis.mean(axis=0) * ls[2:] + lm[2:]

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    names = [("omega_m", r"$\Omega_m$"), ("s8", r"$S_8$")]
    for j, (name, label) in enumerate(names):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(truth[:, j], pred[:, j], s=8, alpha=0.35)
        lo = min(truth[:, j].min(), pred[:, j].min())
        hi = max(truth[:, j].max(), pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}: prediction vs truth")
        fig.tight_layout()
        fig.savefig(out / f"{args.tag}_{name}_pred_vs_true.png", dpi=180)
        plt.close(fig)

    raw_cov = [np.mean(np.abs(pred[:, j] - truth[:, j]) <= sigma_raw[:, j]) for j in range(2)]
    cal_cov = [np.mean(np.abs(pred[:, j] - truth[:, j]) <= sigma_cal[:, j]) for j in range(2)]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, raw_cov, width=0.36, label="raw")
    ax.bar(x + 0.18, cal_cov, width=0.36, label="calibrated")
    ax.axhline(0.68, linestyle="--", linewidth=1, label="nominal 0.68")
    ax.set_xticks(x, ["Omega_m", "S8"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("68% uncertainty interval coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{args.tag}_coverage68.png", dpi=180)
    plt.close(fig)

    nuisance_names = [("tagn", r"$T_{AGN}$"), ("f0", r"$f_0$"), ("delta_z", r"$\Delta z$")]
    for j, (name, label) in enumerate(nuisance_names):
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.scatter(truth[:, j + 2], nuis_pred[:, j], s=8, alpha=0.35)
        lo = min(truth[:, j + 2].min(), nuis_pred[:, j].min())
        hi = max(truth[:, j + 2].max(), nuis_pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"Auxiliary nuisance prediction: {label}")
        fig.tight_layout()
        fig.savefig(out / f"{args.tag}_{name}_pred_vs_true.png", dpi=180)
        plt.close(fig)

    # One loss curve per member.
    for p in sorted((ROOT / "results").glob(f"{args.tag}_history_member_*.json")):
        with p.open(encoding="utf-8") as f:
            h = json.load(f)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label="train")
        ax.plot([r["epoch"] for r in h], [r["val_loss"] for r in h], label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("multitask loss")
        ax.set_title(p.stem.replace("_history_", " - "))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{p.stem}.png", dpi=180)
        plt.close(fig)

    print("wrote inference/calibration figures to", out)


if __name__ == "__main__":
    main()
