"""Utility to inject salt-and-pepper noise for benchmarking."""

from __future__ import annotations

from typing import Optional

import numpy as np


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return a noisy copy of the input image."""

    if not 0 <= amount <= 1:
        raise ValueError("amount must lie in [0, 1]")
    if not 0 <= salt_vs_pepper <= 1:
        raise ValueError("salt_vs_pepper must lie in [0, 1]")

    arr = np.asarray(image).copy()
    if amount == 0:
        return arr

    rng = np.random.default_rng(seed)
    mask = rng.random(arr.shape[:2])
    salt_threshold = amount * salt_vs_pepper
    pepper_threshold = amount

    salt_mask = mask < salt_threshold
    pepper_mask = (mask >= salt_threshold) & (mask < pepper_threshold)

    if arr.ndim == 2:
        arr[salt_mask] = 255
        arr[pepper_mask] = 0
    else:
        arr[salt_mask, :] = 255
        arr[pepper_mask, :] = 0
    return arr
