"""Benchmark CLI comparing brute-force vs optimized median filtering."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .filters import brute_force_median, optimized_median_filter
from .metrics import psnr
from .noise import add_salt_pepper_noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to source image file")
    parser.add_argument("--kernel", type=int, default=5, help="Odd kernel size (default: 5)")
    parser.add_argument("--noise", type=float, default=0.1, help="Salt-and-pepper noise ratio (0-1)")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timing repetitions")
    parser.add_argument("--output", type=Path, default=Path("reports"), help="Output directory for artifacts")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib visualization")
    args = parser.parse_args()

    suite = BenchmarkSuite(kernel=args.kernel, noise=args.noise, repeats=args.repeats, output_dir=args.output)
    suite.run(args.image, save_plot=not args.no_plot)


@dataclass
class BenchmarkRecord:
    name: str
    elapsed: float
    psnr_value: float
    image: np.ndarray


class BenchmarkSuite:
    """Coordinates benchmarking workflows for the CLI."""

    def __init__(self, *, kernel: int, noise: float, repeats: int, output_dir: Path) -> None:
        self.kernel = kernel
        self.noise = noise
        self.repeats = repeats
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_path: Path, *, save_plot: bool = True) -> None:
        clean = self._load_image(image_path)
        noisy = add_salt_pepper_noise(clean, amount=self.noise)

        records = [
            self._benchmark_case("Brute-force", brute_force_median, noisy, clean),
            self._benchmark_case("Optimized", optimized_median_filter, noisy, clean),
        ]
        self._print_summary(records)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._save_results(records, timestamp, noisy, image_path)
        if save_plot:
            self._plot(records, timestamp)

    # Internal helpers -------------------------------------------------
    def _benchmark_case(
        self,
        name: str,
        func: Callable[[np.ndarray, int], np.ndarray],
        noisy: np.ndarray,
        clean: np.ndarray,
    ) -> BenchmarkRecord:
        timings: List[float] = []
        output = None
        for _ in range(self.repeats):
            start = time.perf_counter()
            candidate = func(noisy, self.kernel)
            timings.append(time.perf_counter() - start)
            if output is None:
                output = candidate
        assert output is not None
        score = psnr(clean, output)
        return BenchmarkRecord(name=name, elapsed=float(np.mean(timings)), psnr_value=score, image=output)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img.convert("RGB" if img.mode != "L" else "L"))

    def _print_summary(self, records: Iterable[BenchmarkRecord]) -> None:
        print("Benchmark Results")
        for record in records:
            print(f"- {record.name:<12} time={record.elapsed:.4f}s  PSNR={record.psnr_value:.2f} dB")

    def _save_results(
        self,
        records: Iterable[BenchmarkRecord],
        timestamp: str,
        noisy: np.ndarray,
        source: Path,
    ) -> None:
        meta = {
            "source_image": str(source),
            "kernel": self.kernel,
            "noise": self.noise,
            "repeats": self.repeats,
            "records": [
                {"name": r.name, "elapsed": r.elapsed, "psnr": r.psnr_value}
                for r in records
            ],
        }
        json_path = self.output_dir / f"benchmark_{timestamp}.json"
        json_path.write_text(json.dumps(meta, indent=2))

        noisy_path = self.output_dir / f"noisy_{timestamp}.png"
        self._save_image(noisy, noisy_path)
        for r in records:
            path = self.output_dir / f"{r.name.lower().replace(' ', '_')}_{timestamp}.png"
            self._save_image(r.image, path)

    def _plot(self, records: List[BenchmarkRecord], timestamp: str) -> None:
        names = [r.name for r in records]
        times = [r.elapsed for r in records]
        scores = [r.psnr_value for r in records]

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].bar(names, times, color=["#999999", "#d33f49"])
        axes[0].set_title("Runtime (s)")
        axes[1].bar(names, scores, color=["#999999", "#d33f49"])
        axes[1].set_title("PSNR (dB)")
        fig.tight_layout()
        plot_path = self.output_dir / f"benchmark_{timestamp}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _save_image(array: np.ndarray, path: Path) -> None:
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        image.save(path)


if __name__ == "__main__":  # pragma: no cover
    main()
