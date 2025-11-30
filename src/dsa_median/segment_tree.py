"""Fenwick tree implementation for intensity histogram queries."""

from __future__ import annotations


class FenwickTree:
    """Supports prefix/range sums in O(log n)."""

    def __init__(self, size: int = 256) -> None:
        self.size = size
        self.tree = [0] * (self.size + 2)

    def clear(self) -> None:
        self.tree = [0] * (self.size + 2)

    def update(self, index: int, delta: int) -> None:
        if index < 0:
            return
        idx = index + 1
        while idx < len(self.tree):
            self.tree[idx] += delta
            idx += idx & -idx

    def prefix_sum(self, index: int) -> int:
        if index < 0:
            return 0
        idx = min(index + 1, self.size + 1)
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx
        return total

    def range_sum(self, left: int, right: int) -> int:
        if right < left:
            return 0
        right = min(right, self.size)
        left = max(left, 0)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
