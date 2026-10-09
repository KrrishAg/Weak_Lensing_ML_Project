#!/usr/bin/env python
"""Print a compact report table comparing measured baseline/proposed runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["baseline", "proposed"])
    args = ap.parse_args()
    rows = []
    for tag in args.tags:
        p = ROOT / "results" / f"{tag}_metrics.json"
        if p.exists():
            rows.append((tag, read(p)))
    if not rows:
        print("No measured results yet. Train the requested runs first.")
        return

    print("| model | nuisance wt | M | Om MAE | S8 MAE | raw cov68 | calibrated cov68 | TAGN MAE | f0 MAE | dz MAE |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, pack in rows:
        raw = pack["test_raw"]
        cal = pack["test_calibrated"] or raw
        print(
            f"| {tag} | {pack['nuisance_weight']:.3f} | {pack['ensemble_size']} | "
            f"{cal['mae_omega_m']:.5f} | {cal['mae_s8']:.5f} | {raw['coverage68_mean']:.3f} | "
            f"{cal['coverage68_mean']:.3f} | " + (f"{cal['mae_tagn']:.4f} | {cal['mae_f0']:.5f} | {cal['mae_delta_z']:.5f} |" if pack['nuisance_weight'] > 0 else "- | - | - |")
        )


if __name__ == "__main__":
    main()
