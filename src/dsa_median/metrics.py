"""Image quality metrics for benchmarking results."""

from __future__ import annotations

import numpy as np


def psnr(reference: np.ndarray, test: np.ndarray, *, max_value: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio in decibels."""

    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)
    mse = np.mean((ref - tst) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_value**2) / mse)
