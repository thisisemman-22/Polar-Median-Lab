import numpy as np

from dsa_median.filters import brute_force_median, optimized_median_filter


def test_heap_backend_matches_on_simple_pattern():
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    brute = brute_force_median(image, 3)
    optimized = optimized_median_filter(image, 3, backend="heap")
    np.testing.assert_allclose(optimized, brute, atol=1)


def test_vectorized_backend_matches_brute_force_on_random_image():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    brute = brute_force_median(image, 5)
    optimized = optimized_median_filter(image, 5, backend="vectorized")
    np.testing.assert_array_equal(optimized, brute)
