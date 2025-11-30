import pytest

from dsa_median.dual_heap import DualHeap


def test_dual_heap_running_median():
    heap = DualHeap()
    values = [5, 1, 9, 3, 8]
    medians = []
    for value in values:
        heap.insert(value)
        medians.append(heap.median())
    assert medians[-1] == 5

    heap.erase(9)
    assert pytest.approx(heap.median()) == 4
