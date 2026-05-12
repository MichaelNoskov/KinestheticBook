"""Datasets for gesture folders plus root-level noise CSV files."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.dataset.csv_read import read_normalized_csv_rows

from .augment import augment_sequence
from .temporal import align_series_to_uniform_grid

FEATURE_KEYS = ("pitch", "roll", "yaw", "dp", "dr", "dy")
NOISE_LABEL = "__noise__"


def _read_csv_timeseries(path: Path, fallback_dt: float) -> tuple[np.ndarray, np.ndarray] | None:
    rows = read_normalized_csv_rows(path)
    if not rows:
        return None

    has_t = "t_mono" in rows[0]
    ts: list[float] = []
    ori: list[tuple[float, float, float]] = []

    for r in rows:
        try:
            p = float(r["pitch"])
            rl = float(r["roll"])
            y = float(r["yaw"])
        except (KeyError, ValueError):
            continue
        if has_t:
            try:
                ts.append(float(r["t_mono"]))
            except (KeyError, ValueError):
                continue
        ori.append((p, rl, y))

    if len(ori) < 2:
        return None

    ori_arr = np.asarray(ori, dtype=np.float32)
    if has_t and len(ts) == len(ori_arr):
        t_arr = np.asarray(ts, dtype=np.float64)
        t_arr = t_arr - t_arr[0]
    else:
        t_arr = np.arange(len(ori_arr), dtype=np.float64) * float(fallback_dt)

    return t_arr, ori_arr


@dataclass(frozen=True)
class SampleMeta:
    path: str
    label: str


class GestureNoiseDataset(Dataset):
    """Uniform-time gesture samples. Folder names become labels."""

    def __init__(
        self,
        data_root: Path,
        seq_len: int = 64,
        grid_dt: float = 0.05,
        min_raw_rows: int = 6,
        noise_window_rows: int = 48,
        noise_stride: int = 24,
        max_noise_samples: int | None = 400,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.seq_len = seq_len
        self.grid_dt = float(grid_dt)
        self.min_raw_rows = min_raw_rows
        self.noise_window_rows = max(noise_window_rows, min_raw_rows)
        self.noise_stride = max(1, noise_stride)
        self.max_noise_samples = max_noise_samples
        self.rng = random.Random(seed + 11)

        self.label_to_idx: dict[str, int] = {}
        self.samples: list[tuple[np.ndarray, int, SampleMeta]] = []

        self._load_gesture_folders()
        self._load_root_noise_csvs()

        if not self.samples:
            raise ValueError(f"Нет ни одной последовательности в {self.data_root}")

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def _ensure_label(self, name: str) -> int:
        if name not in self.label_to_idx:
            idx = len(self.label_to_idx)
            self.label_to_idx[name] = idx
        return self.label_to_idx[name]

    def _vector_from_series(self, t: np.ndarray, ori: np.ndarray) -> np.ndarray | None:
        if len(t) < self.min_raw_rows or len(ori) < self.min_raw_rows:
            return None
        try:
            return align_series_to_uniform_grid(t, ori, self.seq_len, self.grid_dt)
        except ValueError:
            return None

    def _add_series(self, t: np.ndarray, ori: np.ndarray, label: str, meta: SampleMeta) -> None:
        seq = self._vector_from_series(t, ori)
        if seq is None:
            return
        y_idx = self._ensure_label(label)
        self.samples.append((seq, y_idx, meta))

    def _load_gesture_folders(self) -> None:
        if not self.data_root.is_dir():
            return
        for sub in sorted(self.data_root.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            label = sub.name
            for csv_path in sorted(sub.glob("*.csv")):
                ts = _read_csv_timeseries(csv_path, self.grid_dt)
                if ts is None:
                    continue
                t, ori = ts
                self._add_series(t, ori, label, SampleMeta(path=str(csv_path), label=label))

    def _load_root_noise_csvs(self) -> None:
        if not self.data_root.is_dir():
            return
        noise_label = NOISE_LABEL

        windows: list[tuple[np.ndarray, np.ndarray, SampleMeta]] = []
        for csv_path in sorted(self.data_root.glob("*.csv")):
            if not csv_path.is_file():
                continue
            ts = _read_csv_timeseries(csv_path, self.grid_dt)
            if ts is None:
                continue
            t_full, ori_full = ts
            if len(t_full) < self.noise_window_rows:
                continue
            w, st = self.noise_window_rows, self.noise_stride
            for start in range(0, len(t_full) - w + 1, st):
                t_win = t_full[start : start + w].copy()
                ori_win = ori_full[start : start + w].copy()
                t_win = t_win - t_win[0]
                windows.append(
                    (t_win, ori_win, SampleMeta(path=f"{csv_path}#{start}", label=noise_label))
                )

        if self.max_noise_samples is not None and len(windows) > self.max_noise_samples:
            self.rng.shuffle(windows)
            windows = windows[: self.max_noise_samples]

        for t_win, ori_win, meta in windows:
            self._add_series(t_win, ori_win, noise_label, meta)

    def num_classes(self) -> int:
        return len(self.label_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, y, _meta = self.samples[i]
        return torch.from_numpy(seq.astype(np.float32)), torch.tensor(y, dtype=torch.long)

    def export_label_map(self, path: Path) -> None:
        path.write_text(json.dumps(self.label_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")


def stratified_split(
    dataset: GestureNoiseDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, (_s, y, _m) in enumerate(dataset.samples):
        by_label.setdefault(y, []).append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _y, idxs in by_label.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs)
            continue
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        n_val = min(n_val, len(idxs) - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    if not val_idx and len(train_idx) > 1:
        val_idx.append(train_idx.pop())
        rng.shuffle(train_idx)
    return train_idx, val_idx


class SubsetDataset(Dataset):
    """Indexed subset with optional train-time augmentation."""

    def __init__(self, base: GestureNoiseDataset, indices: list[int], *, augment: bool) -> None:
        self.base = base
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, y, _ = self.base.samples[self.indices[i]]
        x = seq.astype(np.float32)
        if self.augment:
            x = augment_sequence(x, training=True)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)


def compute_norm_stats(dataset: GestureNoiseDataset, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    sum_c = np.zeros(len(FEATURE_KEYS), dtype=np.float64)
    sumsq_c = np.zeros(len(FEATURE_KEYS), dtype=np.float64)
    count = 0
    for i in indices:
        seq, _y, _m = dataset.samples[i]
        sum_c += seq.sum(axis=0)
        sumsq_c += (seq.astype(np.float64) ** 2).sum(axis=0)
        count += seq.shape[0]
    mean = (sum_c / max(count, 1)).astype(np.float32)
    var = (sumsq_c / max(count, 1)) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


class NormalizedSubset(Dataset):
    def __init__(self, base: SubsetDataset, mean: np.ndarray, std: np.ndarray) -> None:
        self.base = base
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[i]
        x = (x - self.mean) / self.std
        return x, y
