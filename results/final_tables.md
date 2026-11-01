# Measured final tables

Only completed runs are shown; missing experiments are not imputed.

## Cosmological inference and calibration

| Run | M | nuisance wt | MAE Om | MAE S8 | RMSE Om | RMSE S8 | raw cov68 | calibrated cov68 | Phase-1 ref. score | TAGN MAE | f0 MAE | dz MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (realization split) | 1 | 0.000 | 0.02604 | 0.02270 | 0.03559 | 0.02861 | 0.726 | 0.686 | 10.0181 | - | - | - |
| Proposed M=3 (realization split) | 3 | 0.350 | 0.02583 | 0.02224 | 0.03516 | 0.02784 | 0.733 | 0.681 | 10.1803 | 0.3485 | 0.00706 | 0.01660 |

## Offline controlled-shift OoD evaluation

These are diagnostic synthetic shifts, not the hidden competition OoD distribution.
`nuisance_prior` / `hybrid` use the fixed reference prior bounds and weights [1,1,1]; `nuisance_prior_tuned` / `hybrid_tuned` use prior bounds and fusion weights selected by grid search on validation shifts only (see below).

| Detector | extra_noise | smooth | amplitude | highfreq | Mean low-FPR TPR |
|---|---:|---:|---:|---:|---:|
| ensemble_disagreement | 0.036 | 0.507 | 0.210 | 0.982 | 0.434 |
| nuisance_prior | 0.050 | 0.029 | 0.933 | 1.000 | 0.503 |
| nuisance_prior_tuned | 0.049 | 0.996 | 0.931 | 1.000 | 0.744 |
| power_spectrum | 0.282 | 1.000 | 0.955 | 1.000 | 0.809 |
| hybrid | 0.202 | 1.000 | 0.985 | 1.000 | 0.797 |
| hybrid_tuned | 0.277 | 1.000 | 0.968 | 1.000 | 0.811 |

## Nuisance-prior bound tuning (validation only)

Grid search over 625 candidate windows for (TAGN lo, TAGN hi, f0 lo, f0 hi). Objective: nuisance-prior score alone: mean low-FPR TPR over validation shifts (tie-break mean AUROC).

| Bounds [tagn_lo, tagn_hi, f0_lo, f0_hi] | val mean low-FPR TPR | val mean AUROC |
|---|---:|---:|
| reference [7.2, 8.5, 0.0, 0.0265] | 0.501 | 0.855 |
| tuned [7.6, 8.1, 0.012, 0.016] | 0.744 | 0.890 |

## Fusion-weight hyperparameter tuning (validation only)

Grid search over 231 non-negative weight triplets (step 0.05, normalized to sum 1), using the tuned prior bounds [7.6, 8.1, 0.012, 0.016]. Objective: mean low-FPR TPR over validation shifts (tie-break mean AUROC).

| Weights [disagreement, prior, power spectrum] | val mean low-FPR TPR | val mean AUROC |
|---|---:|---:|
| fixed [1.0, 1.0, 1.0] | 0.827 | 0.976 |
| tuned [0.2, 0.1, 0.7] | 0.860 | 0.987 |

