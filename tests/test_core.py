import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wl_project.data import (
    LabelScaler,
    OfficialWeakLensingDataset,
    PooledCacheWeakLensingDataset,
    average_pool_np,
    make_splits,
    pooled_observed_counts,
    shape_noise_sigma,
)
from wl_project.losses import multitask_loss
from wl_project.metrics import fit_sigma_scale, low_fpr_tpr_score
from wl_project.models import MultiTaskWeakLensingNet
from wl_project.ood import nuisance_prior_violation
from wl_project.physics import radial_power_spectrum


def test_model_shapes():
    m = MultiTaskWeakLensingNet(base_channels=8)
    x = torch.randn(4, 1, 64, 32)
    out = m(x)
    assert out[0].shape == (4, 2)
    assert out[1].shape == (4, 2)
    assert out[2].shape == (4, 3)
    assert torch.all(out[1] > 0)
    y = torch.randn(4, 5)
    loss, _ = multitask_loss(out, y)
    assert torch.isfinite(loss)


def test_power_spectrum_shape():
    assert radial_power_spectrum(np.random.randn(64, 32), 12).shape == (12,)


def test_prior_violation_orders_shift():
    a = nuisance_prior_violation(np.array([[7.8, 0.01, 0.0]]))[0]
    b = nuisance_prior_violation(np.array([[9.0, 0.08, 0.10]]))[0]
    assert b > a


def test_low_fpr_metric():
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[np.zeros(1000), np.ones(1000)]
    assert low_fpr_tpr_score(y, s) > 0.99


def test_cosmology_split_has_no_cosmology_overlap():
    labels = np.zeros((12, 8, 5), dtype=np.float32)
    train, val, test, _, info = make_splits(
        labels, strategy="cosmology", val_cosmologies=2, test_cosmologies=2, seed=1
    )
    tc = {c for c, _ in train}
    vc = {c for c, _ in val}
    xc = {c for c, _ in test}
    assert not (tc & vc or tc & xc or vc & xc)
    assert info["n_train"] == len(train)


def test_sigma_calibration_hits_target_on_calibration_data():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(5000, 2))
    mean = np.zeros_like(y)
    sigma = np.full_like(y, 0.5)
    scale = fit_sigma_scale(y, mean, sigma, target_coverage=0.68)
    cov = np.mean(np.abs(y - mean) <= sigma * scale[None, :], axis=0)
    assert np.all(np.abs(cov - 0.68) < 0.01)


def test_cache_matches_direct_noiseless_pooling(tmp_path):
    mask = np.zeros((8, 4), dtype=bool)
    mask[1:7, 1:4] = True
    n_obs = int(mask.sum())
    labels = np.zeros((2, 3, 5), dtype=np.float32)
    compact = np.arange(2 * 3 * n_obs, dtype=np.float32).reshape(2, 3, n_obs) / 100
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "kappa.npy", compact.astype(np.float16))

    cache = np.empty((2, 3, 4, 2), dtype=np.float16)
    for c in range(2):
        for s in range(3):
            dense = np.zeros(mask.shape, np.float32)
            dense[mask] = compact[c, s]
            cache[c, s] = average_pool_np(dense, 2)
    np.save(tmp_path / "cache.npy", cache)

    scaler = LabelScaler.fit(labels.reshape(-1, 5))
    idx = [(0, 0), (1, 2)]
    direct = OfficialWeakLensingDataset(
        tmp_path,
        idx,
        scaler,
        pool_factor=2,
        add_noise=False,
        kappa_file="kappa.npy",
        label_file="labels.npy",
        mask_file="mask.npy",
    )
    cached = PooledCacheWeakLensingDataset(
        tmp_path / "cache.npy",
        tmp_path,
        idx,
        scaler,
        pool_factor=2,
        add_noise=False,
        label_file="labels.npy",
        mask_file="mask.npy",
    )
    for i in range(len(idx)):
        xd, _ = direct[i]
        xc, _ = cached[i]
        assert np.allclose(xd.numpy(), xc.numpy(), atol=2e-3)


def test_pooled_noise_variance_formula():
    mask = np.ones((8, 8), dtype=bool)
    counts = pooled_observed_counts(mask, 4)
    native = shape_noise_sigma()
    pooled = native * np.sqrt(counts) / 16
    assert np.allclose(pooled, native / 4)
