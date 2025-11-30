from dsa_median.segment_tree import FenwickTree


def test_fenwick_tree_range_sum():
    tree = FenwickTree(10)
    for idx in range(5):
        tree.update(idx, idx + 1)
    assert tree.prefix_sum(4) == 15
    assert tree.range_sum(2, 4) == 12
