#!/usr/bin/env python
"""Run the complete college-project experiment matrix after full data are present.

Experiment A (primary): realization split, baseline vs proposed ensemble.
Experiment B (secondary): held-out cosmology split, proposed ensemble only.
OoD stress tests are run on the primary proposed ensemble.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*parts: object) -> None:
    cmd = [sys.executable, *map(str, parts)]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--ood-n", type=int, default=512)
    args = ap.parse_args()

    primary = ROOT / "configs" / "default.yaml"
    heldout = ROOT / "configs" / "heldout_cosmology.yaml"

    run(ROOT / "scripts" / "inspect_data.py", "--config", primary)
    if not args.skip_cache:
        run(ROOT / "scripts" / "build_cache.py", "--config", primary)

    # Controlled primary ablation: same architecture/data, nuisance supervision off vs on.
    run(ROOT / "scripts" / "train_ensemble.py", "--config", primary,
        "--tag", "baseline_realization", "--ensemble-size", "1", "--nuisance-weight", "0")
    run(ROOT / "scripts" / "make_figures.py", "--tag", "baseline_realization")

    run(ROOT / "scripts" / "train_ensemble.py", "--config", primary,
        "--tag", "proposed_realization", "--ensemble-size", "3", "--nuisance-weight", "0.35")
    run(ROOT / "scripts" / "make_figures.py", "--tag", "proposed_realization")

    run(ROOT / "scripts" / "compare_ablation.py", "--tags",
        "baseline_realization", "proposed_realization")

    # Offline OoD stress test on the primary proposed model.
    run(ROOT / "scripts" / "evaluate_ood.py", "--config", primary,
        "--tag", "proposed_realization", "--n", str(args.ood_n))
    run(ROOT / "scripts" / "summarize_ood.py", "--tag", "proposed_realization")
    run(ROOT / "scripts" / "make_ood_figures.py", "--tag", "proposed_realization")

    # Secondary generalization experiment with completely unseen cosmologies.
    run(ROOT / "scripts" / "train_ensemble.py", "--config", heldout,
        "--tag", "proposed_heldout_cosmology", "--ensemble-size", "3", "--nuisance-weight", "0.35")
    run(ROOT / "scripts" / "make_figures.py", "--tag", "proposed_heldout_cosmology")

    run(ROOT / "scripts" / "make_final_tables.py")


if __name__ == "__main__":
    main()
