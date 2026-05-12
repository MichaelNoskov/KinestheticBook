"""Uniform time grid and forward-fill features."""
from __future__ import annotations

import numpy as np


def orientation_grid_to_features(grid_ori: np.ndarray) -> np.ndarray:
    grid_ori = np.asarray(grid_ori, dtype=np.float32)
    t_steps, three = grid_ori.shape
    if three != 3:
        raise ValueError("orientation must be (T, 3)")
    out = np.zeros((t_steps, 6), dtype=np.float32)
    prev = None
    for i in range(t_steps):
        p, r, y = float(grid_ori[i, 0]), float(grid_ori[i, 1]), float(grid_ori[i, 2])
        if prev is None:
            dp = dr = dy = 0.0
        else:
            pp, prr, py = prev
            dp, dr, dy = p - pp, r - prr, y - py
        out[i] = (p, r, y, dp, dr, dy)
        prev = (p, r, y)
    return out


def align_series_to_uniform_grid(
    t: np.ndarray,
    orientation: np.ndarray,
    seq_len: int,
    dt: float,
) -> np.ndarray:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if dt <= 0:
        raise ValueError("dt must be positive")

    t = np.asarray(t, dtype=np.float64).reshape(-1)
    ori = np.asarray(orientation, dtype=np.float32).reshape(-1, 3)
    if len(t) != len(ori) or len(t) < 2:
        raise ValueError("t and orientation must have same length >= 2")

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    ori = ori[order]

    t_end = float(t[-1])
    t_start = t_end - (seq_len - 1) * float(dt)
    grid_t = t_start + np.arange(seq_len, dtype=np.float64) * float(dt)

    grid_ori = np.zeros((seq_len, 3), dtype=np.float32)
    j = 0
    last = ori[0].copy()

    for i in range(seq_len):
        tg = grid_t[i]
        while j < len(t) and t[j] <= tg:
            last = ori[j]
            j += 1
        grid_ori[i] = last

    return orientation_grid_to_features(grid_ori)
