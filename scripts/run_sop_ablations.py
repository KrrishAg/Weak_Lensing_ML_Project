#!/usr/bin/env python
"""Run the three explicit SOP ablation families with one ensemble member each."""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(config, tag):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        tmp = f.name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_ensemble.py"),
        "--config",
        tmp,
        "--tag",
        tag,
        "--ensemble-size",
        "1",
    ]
    print("\nRUN", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    Path(tmp).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument(
        "--family",
        choices=["resolution", "noise", "multiscale", "all"],
        default="all",
    )
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    # Ablations should not accidentally read a k=4 cache when k changes.
    base["data"]["use_cache"] = False
    base["training"]["ensemble_size"] = 1

    if args.family in {"resolution", "all"}:
        for k in [2, 4, 8]:
            cfg = copy.deepcopy(base)
            cfg["data"]["pool_factor"] = k
            run(cfg, f"ablation_resolution_k{k}")

    if args.family in {"noise", "all"}:
        for repeats in [1, 2, 4]:
            cfg = copy.deepcopy(base)
            cfg["data"]["noise_repeats_per_epoch"] = repeats
            run(cfg, f"ablation_noise_x{repeats}")

    if args.family in {"multiscale", "all"}:
        for flag in [False, True]:
            cfg = copy.deepcopy(base)
            cfg["model"]["multiscale"] = flag
            run(cfg, f"ablation_multiscale_{'on' if flag else 'off'}")


if __name__ == "__main__":
    main()
