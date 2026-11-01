# Trustworthy Weak-Lensing ML: Joint Cosmological Inference + OoD Detection

End-to-end project based on the FAIR Universe Weak Lensing ML Uncertainty Challenge.

## Research question
Can one compact model retain accurate cosmological inference while becoming more trustworthy by:
1. using the three nuisance labels discarded by the released direct-CNN baseline,
2. validating/calibrating its uncertainty explicitly, and
3. combining neural uncertainty with a physics-consistency signal for OoD detection?

## What this repository implements
- official FAIR-Universe 2026 public-data downloader with MD5 verification
- memory-bounded NumPy memmap loader
- reconstruction from compact observed-pixel storage to the HSC WIDE12H map
- on-the-fly shape-noise augmentation in float32
- optional average-pooling cache (default k=4)
- two evaluation protocols: realization split (primary) and held-out-cosmology split (secondary)
- shared CNN encoder
- heteroscedastic outputs for Omega_m and S8
- auxiliary regression of TAGN, f0, and Delta-z
- deep ensemble support
- validation-only uncertainty calibration and empirical coverage
- ensemble-disagreement OoD score
- nuisance-prior-violation OoD score, with the TAGN/f0 window bounds tuned by grid search on validation shifts (label ranges kept as reference)
- binned power-spectrum physics-consistency score
- z-score fusion of the three novelty signals, with the 3 fusion weights tuned by grid search on validation shifts (fixed `[1,1,1]` kept as reference)
- controlled synthetic distribution shifts for offline OoD evaluation
- automatic report-ready figures, Markdown tables, and CSV tables

## Official dataset used
The main experiments use the upgraded 2026 Zenodo release:

- `WIDE12H_bin2_2arcmin_kappa_newrealization.npy`: `(101, 256, 132019)`, float16
- `label_newrealization.npy`: `(101, 256, 5)`
- `WIDE12H_bin2_2arcmin_mask.npy`: `(1424, 176)`, bool

The five labels are, in order: `Omega_m, S8, T_AGN, f0, Delta_z`.
A ~23 MB official sample is used first to validate the pipeline before the full ~6.8 GB map file.

## Why there are two split protocols
`configs/default.yaml` is the **primary controlled comparison**. It holds out nuisance realizations but keeps all cosmologies represented across train/validation/test. This is the right setting for asking whether adding nuisance supervision improves the baseline under the same cosmological support.

`configs/heldout_cosmology.yaml` is a **secondary generalization stress test**. Entire cosmologies are withheld from training, so the model must interpolate/generalize to unseen `(Omega_m, S8)` combinations.

Do not merge the two scores into one claim; they answer different questions.

## 0. Environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## 1. Download and inspect the real sample
```bash
python scripts/download_data.py --sample
python scripts/inspect_data.py --config configs/sample.yaml
python scripts/train_ensemble.py --config configs/sample.yaml --tag sample_check --ensemble-size 1
```

## 2. Download the full official training set
```bash
python scripts/download_data.py --full
python scripts/inspect_data.py --config configs/default.yaml
```

If direct Zenodo downloading is blocked but you already downloaded the files in a browser or mounted Google Drive, import them with checksum verification:
```bash
python scripts/import_local_data.py --full --source-dir /path/to/downloaded/files
python scripts/inspect_data.py --config configs/default.yaml
```
For the sample use `--sample` instead of `--full`.

## 3. Build the k=4 noiseless pooled cache
```bash
python scripts/build_cache.py --config configs/default.yaml
```
Fresh shape noise is still added during training; the cache does not freeze a noisy realization.

## 4. Primary controlled baseline
Same encoder/data/split, nuisance loss disabled:
```bash
python scripts/train_ensemble.py \
  --config configs/default.yaml \
  --tag baseline_realization \
  --ensemble-size 1 \
  --nuisance-weight 0
```

## 5. Primary proposed ensemble
```bash
python scripts/train_ensemble.py \
  --config configs/default.yaml \
  --tag proposed_realization \
  --ensemble-size 3 \
  --nuisance-weight 0.35
```

## 6. Inference figures and controlled ablation
```bash
python scripts/make_figures.py --tag baseline_realization
python scripts/make_figures.py --tag proposed_realization
python scripts/compare_ablation.py --tags baseline_realization proposed_realization
```

## 7. Offline OoD stress test
The shifts are diagnostics only; they are **not** claimed to be the competition's hidden OoD physics.
```bash
python scripts/evaluate_ood.py --config configs/default.yaml --tag proposed_realization --n 512
python scripts/summarize_ood.py --tag proposed_realization
python scripts/make_ood_figures.py --tag proposed_realization
```
With `ood.tune_prior_bounds: true` (default), `evaluate_ood.py` first grid-searches the nuisance-prior
window bounds `[tagn_lo, tagn_hi, f0_lo, f0_hi]` over `ood.prior_grid` on **validation** maps and
their shifted copies only, maximizing the prior score's mean low-FPR TPR. The result is written to
`results/proposed_realization_prior_tuning.json` and reported on test as `nuisance_prior_tuned`.

With `ood.tune_fusion_weights: true` (default), `evaluate_ood.py` then grid-searches the three fusion
weights `[disagreement, nuisance prior, power spectrum]` on the simplex (step `ood.fusion_grid_step`)
using **validation** maps and their shifted copies only, maximizing mean low-FPR TPR. The selected
weights are written to `results/proposed_realization_fusion_tuning.json` and evaluated once on the
untouched test split as `hybrid_tuned`, next to the fixed-weight `hybrid`.

## 8. Secondary held-out-cosmology experiment
```bash
python scripts/train_ensemble.py \
  --config configs/heldout_cosmology.yaml \
  --tag proposed_heldout_cosmology \
  --ensemble-size 3 \
  --nuisance-weight 0.35
python scripts/make_figures.py --tag proposed_heldout_cosmology
```

## 9. Generate final report tables
```bash
python scripts/make_final_tables.py
```
Outputs:
- `results/final_tables.md`
- `results/final_inference_table.csv`
- `results/final_ood_table.csv` when OoD evaluation has been run

## 10. One-command complete experiment matrix
After the full data are downloaded:
```bash
python scripts/run_all_experiments.py
```
Use `--skip-cache` if the k=4 cache already exists.

## Optional SOP ablations
```bash
python scripts/run_sop_ablations.py --family resolution
python scripts/run_sop_ablations.py --family noise
python scripts/run_sop_ablations.py --family multiscale
```
These are lower priority than obtaining the primary baseline/proposed result and the held-out-cosmology result.

## What not to claim
- The controlled synthetic shifts are not the hidden competition OoD distribution.
- The compact CNN here is a controlled baseline inspired by the released direct-prediction baseline; do not claim an exact reproduction of the published 8.52 score unless you separately execute the organizer notebook under its original protocol.
- Do not report unmeasured numbers.
- Nuisance prediction is auxiliary supervision, not Bayesian marginalization by itself.
- Held-out-cosmology and realization-split metrics answer different scientific questions.