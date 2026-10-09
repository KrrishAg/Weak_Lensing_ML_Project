# Data directory

The project uses the upgraded FAIR Universe public weak-lensing dataset from Zenodo record `20056065`.

## `data/sample/`
Small official development set:
- `sampled_WIDE12H_bin2_2arcmin_kappa.npy` — `(3, 30, 132019)`, float16
- `sampled_label.npy` — `(3, 30, 5)`
- `WIDE12H_bin2_2arcmin_mask.npy` — `(1424, 176)`, bool

## `data/raw/`
Full training set:
- `WIDE12H_bin2_2arcmin_kappa_newrealization.npy` — `(101, 256, 132019)`, float16
- `label_newrealization.npy` — `(101, 256, 5)`
- `WIDE12H_bin2_2arcmin_mask.npy` — `(1424, 176)`, bool

Use `python scripts/download_data.py --sample` or `--full`.

## `data/cache/`
Optional pooled, noiseless float16 cache. For `k=4`, the full cache is approximately 0.8 GB and has shape `(101, 256, 356, 44)`. Shape noise remains on-the-fly, so the cache does not freeze augmentation.

Raw data are intentionally excluded from Git by `.gitignore`.
