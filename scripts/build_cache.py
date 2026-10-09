#!/usr/bin/env python
"""Build the optional pooled float16 noiseless cache promised in the SOP."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from tqdm import tqdm

from wl_project.data import average_pool_np
from wl_project.utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    d = cfg["data"]
    factor = int(d["pool_factor"])
    data_dir = ROOT / d["data_dir"]
    out = ROOT / d["cache_file"]
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.overwrite:
        print(f"Cache already exists: {out}")
        print("Use --overwrite to rebuild it.")
        return

    compact = np.load(data_dir / d["kappa_file"], mmap_mode="r")
    mask = np.load(data_dir / d["mask_file"], mmap_mode="r").astype(bool)
    n_cosmo, n_sys, n_obs = compact.shape
    if n_obs != int(mask.sum()):
        raise ValueError("Compact data are incompatible with the mask")

    pooled_h = mask.shape[0] // factor
    pooled_w = mask.shape[1] // factor
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    cache = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=np.float16,
        shape=(n_cosmo, n_sys, pooled_h, pooled_w),
    )

    dense = np.zeros(mask.shape, dtype=np.float32)
    for c in tqdm(range(n_cosmo), desc="cosmology"):
        for s in range(n_sys):
            dense.fill(0.0)
            dense[mask] = np.asarray(compact[c, s], dtype=np.float32)
            cache[c, s] = average_pool_np(dense, factor).astype(np.float16)
        cache.flush()

    del cache
    os.replace(tmp, out)
    size_gb = out.stat().st_size / 1024**3
    print(f"Cache written: {out} ({size_gb:.3f} GiB)")


if __name__ == "__main__":
    main()
