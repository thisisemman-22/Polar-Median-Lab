"""Dual-heap data structure for maintaining streaming medians."""

from __future__ import annotations

import heapq
from collections import Counter
from typing import List


class DualHeap:
    """Keeps lower and upper halves of a stream in sync for O(log n) medians."""

    def __init__(self) -> None:
        self.low: List[float] = []  # max-heap implemented via negated values
        self.high: List[float] = []  # min-heap
        self.delayed: Counter[float] = Counter()
        self.low_size = 0
        self.high_size = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.low_size + self.high_size

    def clear(self) -> None:
        """Reset internal heaps."""

        self.low.clear()
        self.high.clear()
        self.delayed.clear()
        self.low_size = 0
        self.high_size = 0

    def insert(self, value: float) -> None:
        if not self.low or value <= -self.low[0]:
            heapq.heappush(self.low, -value)
            self.low_size += 1
        else:
            heapq.heappush(self.high, value)
            self.high_size += 1
        self._rebalance()

    def erase(self, value: float) -> None:
        """Lazy-delete a value. Caller guarantees the value exists."""

        self.delayed[value] += 1
        if self.low and value <= -self.low[0]:
            self.low_size -= 1
            if value == -self.low[0]:
                self._prune(self.low, is_low=True)
        else:
            self.high_size -= 1
            if self.high and value == self.high[0]:
                self._prune(self.high, is_low=False)
        self._rebalance()

    def median(self) -> float:
        if len(self) == 0:
            raise ValueError("Median requested from empty DualHeap")

        self._prune(self.low, is_low=True)
        self._prune(self.high, is_low=False)
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2.0

    # Internal helpers -------------------------------------------------
    def _rebalance(self) -> None:
        if self.low_size > self.high_size + 1:
            value = -heapq.heappop(self.low)
            heapq.heappush(self.high, value)
            self.low_size -= 1
            self.high_size += 1
            self._prune(self.low, is_low=True)
        elif self.high_size > self.low_size:
            value = heapq.heappop(self.high)
            heapq.heappush(self.low, -value)
            self.high_size -= 1
            self.low_size += 1
            self._prune(self.high, is_low=False)

    def _prune(self, heap: List[float], *, is_low: bool) -> None:
        """Remove values that have been lazily deleted."""

        while heap:
            value = -heap[0] if is_low else heap[0]
            if self.delayed.get(value, 0):
                self.delayed[value] -= 1
                if self.delayed[value] == 0:
                    self.delayed.pop(value)
                heapq.heappop(heap)
            else:
                break
