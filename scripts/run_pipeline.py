#!/usr/bin/env python
"""One-command orchestration after the official dataset is present."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(*parts):
    cmd = [sys.executable, *map(str, parts)]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--ood-n", type=int, default=512)
    args = ap.parse_args()

    call(ROOT / "scripts" / "inspect_data.py", "--config", args.config)
    if args.build_cache:
        call(ROOT / "scripts" / "build_cache.py", "--config", args.config)

    if not args.skip_baseline:
        call(
            ROOT / "scripts" / "train_ensemble.py",
            "--config",
            args.config,
            "--tag",
            "baseline",
            "--ensemble-size",
            "1",
            "--nuisance-weight",
            "0",
        )
        call(ROOT / "scripts" / "make_figures.py", "--tag", "baseline")

    call(
        ROOT / "scripts" / "train_ensemble.py",
        "--config",
        args.config,
        "--tag",
        "proposed",
    )
    call(ROOT / "scripts" / "make_figures.py", "--tag", "proposed")
    call(ROOT / "scripts" / "compare_ablation.py")
    call(
        ROOT / "scripts" / "evaluate_ood.py",
        "--config",
        args.config,
        "--tag",
        "proposed",
        "--n",
        str(args.ood_n),
    )
    call(ROOT / "scripts" / "summarize_ood.py", "--tag", "proposed")
    call(ROOT / "scripts" / "make_ood_figures.py", "--tag", "proposed")


if __name__ == "__main__":
    main()
