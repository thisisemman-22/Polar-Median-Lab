"""Sliding window helpers used by both brute-force and optimized filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pad_image(image: np.ndarray, kernel: int, mode: str = "reflect") -> np.ndarray:
    pad = kernel // 2
    if image.ndim == 2:
        pad_width = ((pad, pad), (pad, pad))
    elif image.ndim == 3:
        pad_width = ((pad, pad), (pad, pad), (0, 0))
    else:
        raise ValueError("Expected a 2D grayscale or 3D color image array")
    return np.pad(image, pad_width, mode=mode)


def brute_force_windows(image: np.ndarray, kernel: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield (row, col, block) triples for brute-force median filtering."""

    padded = pad_image(image, kernel)
    window_view = sliding_window_view(padded, (kernel, kernel))
    h, w = window_view.shape[:2]
    for row in range(h):
        for col in range(w):
            yield row, col, window_view[row, col]


@dataclass
class ColumnCache:
    """Caches vertical strips to reduce repeated slicing overhead."""

    kernel: int
    base_row: int
    padded: np.ndarray
    store: Dict[int, np.ndarray]

    def __init__(self, padded: np.ndarray, kernel: int, base_row: int) -> None:
        self.kernel = kernel
        self.base_row = base_row
        self.padded = padded
        self.store = {}

    def get(self, column: int) -> np.ndarray:
        if column not in self.store:
            self.store[column] = self.padded[self.base_row : self.base_row + self.kernel, column].copy()
        return self.store[column]
