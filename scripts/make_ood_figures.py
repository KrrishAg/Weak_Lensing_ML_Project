#!/usr/bin/env python
"""Create report-ready bar charts from the offline OoD stress test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="proposed")
    args = ap.parse_args()
    p = ROOT / "results" / f"{args.tag}_ood_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Run scripts/evaluate_ood.py first: {p}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    shifts = list(data)
    methods = ["ensemble_disagreement", "nuisance_prior", "power_spectrum", "hybrid"]
    labels = ["ensemble", "nuisance prior", "power spectrum", "hybrid"]
    x = np.arange(len(shifts))
    width = 0.18

    for metric, ylabel in [("low_fpr_tpr", "Mean TPR at FPR 0.1%-5%"), ("auroc", "AUROC")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for j, (method, label) in enumerate(zip(methods, labels)):
            vals = [data[s][method][metric] for s in shifts]
            ax.bar(x + (j - 1.5) * width, vals, width=width, label=label)
        ax.set_xticks(x, shifts)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Offline OoD stress test - {metric}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = ROOT / "figures" / f"{args.tag}_ood_{metric}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
