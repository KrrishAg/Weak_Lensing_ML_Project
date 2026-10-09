# Measured final tables

Only completed runs are shown; missing experiments are not imputed.

## Cosmological inference and calibration

| Run | M | nuisance wt | MAE Om | MAE S8 | RMSE Om | RMSE S8 | raw cov68 | calibrated cov68 | Phase-1 ref. score | TAGN MAE | f0 MAE | dz MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (realization split) | 1 | 0.000 | 0.02604 | 0.02270 | 0.03559 | 0.02861 | 0.726 | 0.686 | 10.0181 | - | - | - |
| Proposed M=3 (realization split) | 3 | 0.350 | 0.02583 | 0.02224 | 0.03516 | 0.02784 | 0.733 | 0.681 | 10.1803 | 0.3485 | 0.00706 | 0.01660 |

## Offline controlled-shift OoD evaluation

These are diagnostic synthetic shifts, not the hidden competition OoD distribution.

| Detector | extra_noise | smooth | amplitude | highfreq | Mean low-FPR TPR |
|---|---:|---:|---:|---:|---:|
| ensemble_disagreement | 0.036 | 0.507 | 0.210 | 0.982 | 0.434 |
| nuisance_prior | 0.050 | 0.029 | 0.933 | 1.000 | 0.503 |
| power_spectrum | 0.282 | 1.000 | 0.955 | 1.000 | 0.809 |
| hybrid | 0.202 | 1.000 | 0.985 | 1.000 | 0.797 |

