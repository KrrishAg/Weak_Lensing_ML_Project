from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


LABEL_NAMES = ("omega_m", "s8", "tagn", "f0", "delta_z")


@dataclass
class LabelScaler:
    """Simple per-target standardization fitted only on the training split."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, labels: np.ndarray) -> "LabelScaler":
        labels = np.asarray(labels, dtype=np.float32)
        mean = labels.mean(axis=0).astype(np.float32)
        std = labels.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std + self.mean


def shape_noise_sigma(
    ng: float = 30.0,
    pixel_size_arcmin: float = 2.0,
    sigma_epsilon: float = 0.4,
) -> float:
    """Per-native-pixel shape-noise standard deviation used by the organizers."""

    return float(sigma_epsilon / np.sqrt(2.0 * ng * pixel_size_arcmin**2))


def add_shape_noise(
    x: np.ndarray,
    mask: np.ndarray,
    ng: float = 30.0,
    pixel_size_arcmin: float = 2.0,
    sigma_epsilon: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add independent Gaussian shape noise only on observed pixels."""

    rng = rng or np.random.default_rng()
    sigma = np.float32(shape_noise_sigma(ng, pixel_size_arcmin, sigma_epsilon))
    noise = rng.standard_normal(x.shape, dtype=np.float32) * sigma
    return x.astype(np.float32, copy=False) + noise * mask.astype(np.float32, copy=False)


def average_pool_np(x: np.ndarray, factor: int) -> np.ndarray:
    """Average pool a single 2-D map by an integer factor."""

    if factor == 1:
        return np.asarray(x, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
    y = F.avg_pool2d(t, kernel_size=factor, stride=factor)
    return y[0, 0].numpy()


def pooled_observed_counts(mask: np.ndarray, factor: int) -> np.ndarray:
    """Number of observed native pixels contributing to each pooled cell."""

    if factor == 1:
        return mask.astype(np.float32)
    t = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
    # avg_pool * factor^2 = number of observed pixels in each non-overlapping block.
    return (F.avg_pool2d(t, factor, factor)[0, 0].numpy() * factor**2).astype(np.float32)


def _limit(items: list[tuple[int, int]], n: int | None, rng: np.random.Generator):
    rng.shuffle(items)
    if n is None:
        return items
    return items[: min(int(n), len(items))]


def make_splits(
    labels: np.ndarray,
    strategy: str = "cosmology",
    val_cosmologies: int = 10,
    test_cosmologies: int = 10,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 5566,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
):
    """
    Build train/validation/test splits and fit the label scaler on train only.

    `cosmology` holds out complete cosmological models, which is the preferred
    scientific evaluation for generalization to unseen Omega_m/S8 combinations.
    `realization` reproduces the starter-style split over nuisance realizations.
    """

    n_cosmo, n_sys, n_label = labels.shape
    if n_label != 5:
        raise ValueError(f"Expected five labels, got shape {labels.shape}")

    rng = np.random.default_rng(seed)
    strategy = strategy.lower()

    if strategy == "cosmology":
        if n_cosmo < 3:
            raise ValueError("Cosmology split requires at least 3 cosmologies")
        order = rng.permutation(n_cosmo)
        n_test = min(int(test_cosmologies), max(1, n_cosmo - 2))
        n_val = min(int(val_cosmologies), max(1, n_cosmo - n_test - 1))
        test_cos = sorted(order[:n_test].tolist())
        val_cos = sorted(order[n_test : n_test + n_val].tolist())
        train_cos = sorted(order[n_test + n_val :].tolist())
        train = [(c, s) for c in train_cos for s in range(n_sys)]
        val = [(c, s) for c in val_cos for s in range(n_sys)]
        test = [(c, s) for c in test_cos for s in range(n_sys)]
        split_info = {
            "strategy": "cosmology",
            "train_cosmologies": train_cos,
            "val_cosmologies": val_cos,
            "test_cosmologies": test_cos,
        }
    elif strategy == "realization":
        sys_idx = np.arange(n_sys)
        train_sys, temp_sys = train_test_split(
            sys_idx,
            test_size=float(val_fraction + test_fraction),
            random_state=seed,
        )
        if len(temp_sys) < 2:
            raise ValueError("Not enough realizations for validation/test split")
        test_share = test_fraction / (val_fraction + test_fraction)
        val_sys, test_sys = train_test_split(
            temp_sys,
            test_size=test_share,
            random_state=seed + 1,
        )
        train = [(c, int(s)) for c in range(n_cosmo) for s in train_sys]
        val = [(c, int(s)) for c in range(n_cosmo) for s in val_sys]
        test = [(c, int(s)) for c in range(n_cosmo) for s in test_sys]
        split_info = {
            "strategy": "realization",
            "train_realizations": sorted(map(int, train_sys)),
            "val_realizations": sorted(map(int, val_sys)),
            "test_realizations": sorted(map(int, test_sys)),
        }
    else:
        raise ValueError("split strategy must be 'cosmology' or 'realization'")

    train = _limit(train, max_train_samples, rng)
    val = _limit(val, max_val_samples, rng)
    test = _limit(test, max_test_samples, rng)

    train_labels = np.stack([labels[c, s] for c, s in train]).astype(np.float32)
    scaler = LabelScaler.fit(train_labels)
    split_info.update(
        {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        }
    )
    return train, val, test, scaler, split_info


class _BaseWeakLensingDataset(Dataset):
    def __init__(
        self,
        indices: Iterable[tuple[int, int]],
        labels: np.ndarray,
        label_scaler: LabelScaler,
        add_noise: bool,
        image_mean: float,
        image_std: float,
        ng: float,
        pixel_size_arcmin: float,
        sigma_epsilon: float,
        base_seed: int,
        fresh_noise: bool,
    ):
        self.indices = list(indices)
        self.labels = labels
        self.label_scaler = label_scaler
        self.noise = bool(add_noise)
        self.image_mean = float(image_mean)
        self.image_std = max(float(image_std), 1e-8)
        self.ng = float(ng)
        self.pixel_size_arcmin = float(pixel_size_arcmin)
        self.sigma_epsilon = float(sigma_epsilon)
        self.base_seed = int(base_seed)
        self.fresh_noise = bool(fresh_noise)

    def __len__(self):
        return len(self.indices)

    def _rng(self, i: int) -> np.random.Generator:
        if self.fresh_noise:
            # NumPy is seeded per DataLoader worker in training.worker_init_fn.
            seed = int(np.random.randint(0, 2**32 - 1))
        else:
            seed = (self.base_seed + i * 1009) % (2**32 - 1)
        return np.random.default_rng(seed)

    def _label(self, c: int, s: int) -> torch.Tensor:
        y = np.asarray(self.labels[c, s], dtype=np.float32)
        return torch.from_numpy(self.label_scaler.transform(y).astype(np.float32))


class OfficialWeakLensingDataset(_BaseWeakLensingDataset):
    """Memory-mapped compact-data loader; reconstructs only one dense map at a time."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: Iterable[tuple[int, int]],
        label_scaler: LabelScaler,
        pool_factor: int = 4,
        add_noise: bool = True,
        image_mean: float = 0.0,
        image_std: float = 1.0,
        ng: float = 30.0,
        pixel_size_arcmin: float = 2.0,
        sigma_epsilon: float = 0.4,
        base_seed: int = 0,
        kappa_file: str = "WIDE12H_bin2_2arcmin_kappa_newrealization.npy",
        label_file: str = "label_newrealization.npy",
        mask_file: str = "WIDE12H_bin2_2arcmin_mask.npy",
        fresh_noise: bool = False,
    ):
        data_dir = Path(data_dir)
        labels = np.load(data_dir / label_file, mmap_mode="r")
        super().__init__(
            indices,
            labels,
            label_scaler,
            add_noise,
            image_mean,
            image_std,
            ng,
            pixel_size_arcmin,
            sigma_epsilon,
            base_seed,
            fresh_noise,
        )
        self.data_dir = data_dir
        self.mask = np.load(data_dir / mask_file, mmap_mode="r").astype(bool)
        self.compact = np.load(data_dir / kappa_file, mmap_mode="r")
        self.pool_factor = int(pool_factor)
        self.height, self.width = self.mask.shape

        if self.compact.shape[:2] != labels.shape[:2]:
            raise ValueError(
                f"Kappa/label axes disagree: {self.compact.shape[:2]} vs {labels.shape[:2]}"
            )
        if self.compact.shape[-1] != int(self.mask.sum()):
            raise ValueError(
                f"Compact pixel count {self.compact.shape[-1]} != mask sum {int(self.mask.sum())}"
            )

    def dense_map(self, cosmo_idx: int, sys_idx: int) -> np.ndarray:
        dense = np.zeros((self.height, self.width), dtype=np.float32)
        dense[self.mask] = np.asarray(self.compact[cosmo_idx, sys_idx], dtype=np.float32)
        return dense

    def __getitem__(self, i: int):
        c, s = self.indices[i]
        x = self.dense_map(c, s)
        if self.noise:
            x = add_shape_noise(
                x,
                self.mask,
                self.ng,
                self.pixel_size_arcmin,
                self.sigma_epsilon,
                self._rng(i),
            )
        x = average_pool_np(x, self.pool_factor)
        x = (x - self.image_mean) / self.image_std
        return torch.from_numpy(x[None].astype(np.float32)), self._label(c, s)


class PooledCacheWeakLensingDataset(_BaseWeakLensingDataset):
    """
    Dataset backed by a pooled noiseless float16 cache.

    Noise is still fresh. For a factor-f average, the variance of pooled noise is
    sigma_native^2 * n_observed / f^4, which reproduces adding independent native
    pixel noise before pooling.
    """

    def __init__(
        self,
        cache_file: str | Path,
        data_dir: str | Path,
        indices: Iterable[tuple[int, int]],
        label_scaler: LabelScaler,
        pool_factor: int,
        add_noise: bool = True,
        image_mean: float = 0.0,
        image_std: float = 1.0,
        ng: float = 30.0,
        pixel_size_arcmin: float = 2.0,
        sigma_epsilon: float = 0.4,
        base_seed: int = 0,
        label_file: str = "label_newrealization.npy",
        mask_file: str = "WIDE12H_bin2_2arcmin_mask.npy",
        fresh_noise: bool = False,
    ):
        data_dir = Path(data_dir)
        labels = np.load(data_dir / label_file, mmap_mode="r")
        super().__init__(
            indices,
            labels,
            label_scaler,
            add_noise,
            image_mean,
            image_std,
            ng,
            pixel_size_arcmin,
            sigma_epsilon,
            base_seed,
            fresh_noise,
        )
        self.cache = np.load(cache_file, mmap_mode="r")
        self.pool_factor = int(pool_factor)
        mask = np.load(data_dir / mask_file, mmap_mode="r").astype(bool)
        self.counts = pooled_observed_counts(mask, self.pool_factor)
        expected_hw = self.counts.shape
        if self.cache.shape[:2] != labels.shape[:2] or self.cache.shape[-2:] != expected_hw:
            raise ValueError(
                f"Cache shape {self.cache.shape} is incompatible with labels {labels.shape} "
                f"and pooled map size {expected_hw}"
            )
        native_sigma = shape_noise_sigma(ng, pixel_size_arcmin, sigma_epsilon)
        self.pooled_noise_sigma = (
            native_sigma * np.sqrt(self.counts) / (self.pool_factor**2)
        ).astype(np.float32)

    def __getitem__(self, i: int):
        c, s = self.indices[i]
        x = np.asarray(self.cache[c, s], dtype=np.float32).copy()
        if self.noise:
            noise = self._rng(i).standard_normal(x.shape, dtype=np.float32)
            x += noise * self.pooled_noise_sigma
        x = (x - self.image_mean) / self.image_std
        return torch.from_numpy(x[None].astype(np.float32)), self._label(c, s)


def make_dataset(
    cfg_data: dict,
    root: str | Path,
    indices: Iterable[tuple[int, int]],
    scaler: LabelScaler,
    image_mean: float,
    image_std: float,
    add_noise_flag: bool,
    base_seed: int,
    fresh_noise: bool,
):
    """Factory choosing cache-backed data when requested and available."""

    root = Path(root)
    data_dir = root / cfg_data["data_dir"]
    common = dict(
        indices=indices,
        label_scaler=scaler,
        add_noise=add_noise_flag,
        image_mean=image_mean,
        image_std=image_std,
        ng=cfg_data["ng"],
        pixel_size_arcmin=cfg_data["pixel_size_arcmin"],
        sigma_epsilon=cfg_data.get("sigma_epsilon", 0.4),
        base_seed=base_seed,
        fresh_noise=fresh_noise,
    )
    cache_path = root / cfg_data.get("cache_file", "")
    if cfg_data.get("use_cache", False) and cache_path.exists():
        return PooledCacheWeakLensingDataset(
            cache_file=cache_path,
            data_dir=data_dir,
            pool_factor=cfg_data["pool_factor"],
            label_file=cfg_data["label_file"],
            mask_file=cfg_data["mask_file"],
            **common,
        )
    return OfficialWeakLensingDataset(
        data_dir=data_dir,
        pool_factor=cfg_data["pool_factor"],
        kappa_file=cfg_data["kappa_file"],
        label_file=cfg_data["label_file"],
        mask_file=cfg_data["mask_file"],
        **common,
    )


def estimate_image_stats(dataset: Dataset, n_samples: int = 512) -> tuple[float, float]:
    """Estimate a single global mean/std from training maps only."""

    n = min(len(dataset), int(n_samples))
    if n <= 0:
        raise ValueError("Cannot estimate image statistics from an empty dataset")
    vals = []
    old_mean, old_std = dataset.image_mean, dataset.image_std
    dataset.image_mean, dataset.image_std = 0.0, 1.0
    try:
        for i in range(n):
            x, _ = dataset[i]
            vals.append(x.numpy().ravel())
    finally:
        dataset.image_mean, dataset.image_std = old_mean, old_std
    x = np.concatenate(vals)
    return float(x.mean()), float(max(x.std(), 1e-8))


class SyntheticWeakLensingDataset(Dataset):
    """Fast deterministic toy field generator used only by tests/smoke tests."""

    def __init__(self, n=128, height=64, width=32, seed=0, ood=False):
        self.n = n
        self.h = height
        self.w = width
        self.seed = seed
        self.ood = ood
        rng = np.random.default_rng(seed)
        om = rng.uniform(0.08, 0.62, n)
        s8 = rng.uniform(0.68, 0.95, n)
        tagn = rng.uniform(7.2, 8.5, n)
        f0 = rng.uniform(0, 0.0265, n)
        dz = rng.normal(0, 0.022, n)
        self.labels = np.stack([om, s8, tagn, f0, dz], 1).astype(np.float32)
        self.scaler = LabelScaler.fit(self.labels)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        y = self.labels[i]
        yy, xx = np.mgrid[0 : self.h, 0 : self.w]
        field = 0.8 * y[0] * np.sin(xx / (3 + y[1] * 2))
        field += 0.6 * y[1] * np.cos(yy / (5 + y[0] * 2))
        field += (y[2] - 7.85) * 0.15 * np.sin((xx + yy) / 2.5)
        field += y[3] * 8 * np.cos(xx / 1.7) + y[4] * 4
        field += rng.normal(0, 0.25, (self.h, self.w))
        if self.ood:
            field = 1.35 * field + 0.3 * np.sin(xx * 1.8)
        x = (field - field.mean()) / (field.std() + 1e-6)
        return (
            torch.tensor(x[None], dtype=torch.float32),
            torch.tensor(self.scaler.transform(y), dtype=torch.float32),
        )


class RepeatDataset(Dataset):
    """Repeat a dataset r times per epoch; fresh-noise datasets redraw on each access."""

    def __init__(self, dataset: Dataset, repeats: int = 1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, i):
        return self.dataset[i % len(self.dataset)]
