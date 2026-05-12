"""Small time-series augmentations for gesture windows."""
from __future__ import annotations

import random

import numpy as np


def resample_time_axis(seq: np.ndarray, new_len: int) -> np.ndarray:
    t_old, c = seq.shape[0], seq.shape[1]
    if t_old == new_len:
        return seq.astype(np.float32, copy=False)
    if t_old < 2:
        return np.repeat(seq[:1], new_len, axis=0).astype(np.float32)
    x_old = np.linspace(0.0, 1.0, t_old, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, new_len, dtype=np.float64)
    out = np.empty((new_len, c), dtype=np.float32)
    for j in range(c):
        out[:, j] = np.interp(x_new, x_old, seq[:, j].astype(np.float64)).astype(np.float32)
    return out


def augment_sequence(
    seq: np.ndarray,
    *,
    training: bool,
    noise_std: float = 0.35,
    scale_range: tuple[float, float] = (0.88, 1.12),
    time_warp_range: tuple[float, float] = (0.82, 1.18),
    temporal_shift_max: float = 0.12,
) -> np.ndarray:
    x = np.asarray(seq, dtype=np.float32)
    if not training:
        return x.copy()

    y = x.copy()

    if random.random() < 0.85:
        scales = np.random.uniform(scale_range[0], scale_range[1], size=(1, y.shape[1])).astype(np.float32)
        y *= scales

    if random.random() < 0.9:
        y += (np.random.randn(*y.shape).astype(np.float32) * noise_std)

    if random.random() < 0.65:
        t = y.shape[0]
        factor = float(np.random.uniform(time_warp_range[0], time_warp_range[1]))
        mid = max(4, int(round(t * factor)))
        y = resample_time_axis(y, mid)
        y = resample_time_axis(y, t)

    if random.random() < 0.5 and y.shape[0] > 4:
        max_shift = max(1, int(y.shape[0] * temporal_shift_max))
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            pad = np.repeat(y[:1] if shift > 0 else y[-1:], abs(shift), axis=0)
            if shift > 0:
                y = np.concatenate((pad, y[:-shift]), axis=0)
            else:
                y = np.concatenate((y[-shift:], pad), axis=0)

    return y.astype(np.float32)
