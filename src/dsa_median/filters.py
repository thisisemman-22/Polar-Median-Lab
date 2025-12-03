"""Median filter implementations: brute-force and optimized."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .dual_heap import DualHeap
from .segment_tree import FenwickTree
from .sliding_window import ColumnCache, pad_image

AUTO_HEAP_THRESHOLD = 0  # pixels; defaults to "vectorized" unless caller opts into heap fallback


def brute_force_median(image: np.ndarray, kernel: int) -> np.ndarray:
    """Reference implementation that sorts every kernel-sized window."""

    _validate_kernel(kernel)
    array = np.asarray(image)
    if array.ndim == 2:
        return _slow_pythonic_brute_channel(array, kernel)
    if array.ndim == 3:
        channels = [_slow_pythonic_brute_channel(array[..., idx], kernel) for idx in range(array.shape[2])]
        return np.stack(channels, axis=-1)
    raise ValueError("Expected a 2D grayscale or 3D color image array")


def optimized_median_filter(
    image: np.ndarray,
    kernel: int,
    *,
    pad_mode: str = "reflect",
    backend: str = "auto",
    auto_threshold: int = AUTO_HEAP_THRESHOLD,
) -> np.ndarray:
    """Optimized sliding-window median with backend selection.

    ``backend`` options:

    - ``"auto"`` (default): prefer the fast NumPy/vectorized backend. Pass a
      positive ``auto_threshold`` (in pixels) if you want small images to fall
      back to the heap pipeline to highlight the four data structures.
    - ``"heap"``: explicitly force the Python dual-heap/Fenwick/column-cache
      implementation regardless of image size.
    - ``"vectorized"``: force the NumPy backend for maximum throughput.
    """

    _validate_kernel(kernel)
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("Expected a 2D grayscale or 3D color image array")

    backend_key = backend.lower()
    if backend_key not in {"auto", "heap", "vectorized"}:
        raise ValueError("backend must be one of {'auto', 'heap', 'vectorized'}")

    pixel_count = array.shape[0] * array.shape[1]
    if backend_key == "auto":
        use_heap_backend = auto_threshold > 0 and pixel_count <= auto_threshold
    elif backend_key == "heap":
        use_heap_backend = True
    else:  # vectorized
        use_heap_backend = False

    if array.ndim == 2:
        return (
            _optimized_channel_heap(array, kernel, pad_mode)
            if use_heap_backend
            else _vectorized_median_channel(array, kernel)
        )

    channels = []
    for idx in range(array.shape[2]):
        channel = array[..., idx]
        filtered = (
            _optimized_channel_heap(channel, kernel, pad_mode)
            if use_heap_backend
            else _vectorized_median_channel(channel, kernel)
        )
        channels.append(filtered)
    return np.stack(channels, axis=-1)


# ---------------------------------------------------------------------------
# Brute-force helper


def _slow_pythonic_brute_channel(channel: np.ndarray, kernel: int) -> np.ndarray:
    padded = pad_image(channel, kernel, mode="reflect")
    h, w = channel.shape
    output = np.empty((h, w), dtype=np.float32)
    window_size = kernel * kernel

    for row in range(h):
        for col in range(w):
            values = []
            block = padded[row : row + kernel, col : col + kernel]
            for r in range(kernel):
                for c in range(kernel):
                    values.append(int(block[r, c]))
            _insertion_sort(values)
            output[row, col] = values[window_size // 2]

    return np.clip(output, 0, 255).astype(channel.dtype)


def _insertion_sort(values: List[int]) -> None:
    for idx in range(1, len(values)):
        key = values[idx]
        pos = idx - 1
        while pos >= 0 and values[pos] > key:
            values[pos + 1] = values[pos]
            pos -= 1
        values[pos + 1] = key


# ---------------------------------------------------------------------------
# Optimized helper


def _optimized_channel_heap(channel: np.ndarray, kernel: int, pad_mode: str) -> np.ndarray:
    pad = kernel // 2
    padded = pad_image(channel, kernel, mode=pad_mode).astype(np.uint8, copy=False)  # Data structure 1: padded 2D array tiles
    h, w = channel.shape
    output = np.empty((h, w), dtype=np.float32)
    window_area = kernel * kernel

    for row in range(h):
        dual_heap = DualHeap()  # Data structure 2: min/max-heap pair for streaming median
        histogram = FenwickTree(255)  # Data structure 3: tree-based array (Fenwick) for histogram queries
        column_cache = ColumnCache(padded, kernel, row)  # Data structure 4: dictionary-backed column cache
        _seed_window(dual_heap, histogram, padded[row : row + kernel, 0:kernel])

        for col in range(w):
            center = float(padded[row + pad, col + pad])
            output[row, col] = _fuse_pixel(center, dual_heap, histogram, window_area)
            if col == w - 1:
                continue
            _slide_column(dual_heap, histogram, column_cache, col, kernel)

    clipped = np.clip(output, 0, 255)
    return clipped.astype(channel.dtype)


def _vectorized_median_channel(channel: np.ndarray, kernel: int) -> np.ndarray:
    padded = pad_image(channel, kernel, mode="reflect")
    windows = sliding_window_view(padded, (kernel, kernel))
    flattened = windows.reshape(windows.shape[0], windows.shape[1], -1)
    median_index = flattened.shape[-1] // 2
    partitioned = np.partition(flattened, median_index, axis=-1)
    medians = partitioned[..., median_index]
    return medians.astype(channel.dtype, copy=False)


def _seed_window(heap: DualHeap, hist: FenwickTree, block: np.ndarray) -> None:
    for value in block.ravel():
        _add_sample(heap, hist, int(value))


def _slide_column(heap: DualHeap, hist: FenwickTree, cache: ColumnCache, col: int, kernel: int) -> None:
    outgoing = cache.get(col)
    incoming = cache.get(col + kernel)
    for value in outgoing:
        _remove_sample(heap, hist, int(value))
    for value in incoming:
        _add_sample(heap, hist, int(value))


def _fuse_pixel(center: float, heap: DualHeap, hist: FenwickTree, total: int) -> float:
    """Blend median with center pixel based on histogram saturation."""

    median_value = heap.median()
    saturated = hist.range_sum(0, 8) + hist.range_sum(247, 255)
    ratio = saturated / max(total, 1)
    if ratio > 0.25:
        return median_value
    return 0.7 * median_value + 0.3 * center


def _add_sample(heap: DualHeap, hist: FenwickTree, value: int) -> None:
    heap.insert(value)
    hist.update(_clip_intensity(value), 1)


def _remove_sample(heap: DualHeap, hist: FenwickTree, value: int) -> None:
    heap.erase(value)
    hist.update(_clip_intensity(value), -1)


def _clip_intensity(value: float) -> int:
    return int(min(255, max(0, round(value))))




def _validate_kernel(kernel: int) -> None:
    if kernel % 2 == 0 or kernel < 3:
        raise ValueError("Kernel size must be an odd integer >= 3")
