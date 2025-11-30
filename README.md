# Accelerating Median Filtering for Image Noise Reduction

Course final project for CpE 411 (Data Structures and Algorithms) – Batangas State University TNEU. The system delivers a high-performance median filter for salt-and-pepper noise removal by combining three synergistic data structures: 2D arrays for pixel storage, a priority-queue pair (dual heap) for O(log k) median maintenance, and a tree-based array (Fenwick/segment tree) for fast histogram queries augmented with a dictionary cache. Benchmarks compare the optimized filter against a brute-force baseline to quantify speedups and fidelity.

## Core Requirements
- **Data Structures**: 2D Array, Priority Queue Pair (Min-Max heaps), Fenwick/Segment Tree, Dictionary cache.
- **Algorithms**:
  - Optimized sliding-window median maintenance using dual heaps.
  - Fenwick-tree range-query aggregation for histogram-aware filtering.
  - Baseline brute-force median filtering for comparison.
- **Language/Stack**: Python 3.11+, NumPy, Pillow, Matplotlib, scikit-image.

## Repository Layout
```
.
├── data/
│   └── samples/
├── docs/
│   └── proposal.md
├── src/
│   └── dsa_median/
│       ├── __init__.py
│       ├── dual_heap.py
│       ├── segment_tree.py
│       ├── sliding_window.py
│       ├── filters.py
│       ├── noise.py
│       ├── metrics.py
│       ├── benchmarks.py
│       └── webapp.py
├── tests/
├── pyproject.toml
└── README.md
```

## Quick Start
1. Create and activate a virtual environment (PowerShell example):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   pip install -e .
   ```
3. Run the benchmark CLI to compare filters:
   ```powershell
   python -m dsa_median.benchmarks --image data/samples/lena.png --kernel 5 --noise 0.1
   ```
4. Launch the Flask API (headless HTTP endpoint) if you want to drive the filters over HTTP:
   ```powershell
   python -m flask --app src/dsa_median/webapp.py run --no-reload --port 8000
   ```
   The root route responds with a 404 when no static UI is bundled, but the `/api/denoise` endpoint remains available for programmatic clients.
5. Capture benchmark figures or logs from the CLI output to document acceleration and fidelity.

## Filtering Backends
- `optimized_median_filter(..., backend="heap")` forces the original dual-heap + Fenwick-tree pipeline that demonstrates the required data structures.
- `backend="vectorized"` uses a pure NumPy sliding-window/partition backend that runs in C and is ideal for multi-megapixel frames (what the Flask API uses by default).
- `backend="auto"` (default) picks the heap path for smaller pedagogical samples and switches to the fast vectorized backend once the image exceeds ~320×320 pixels.

## API Notes
- **Endpoint**: `POST /api/denoise` accepts `image` (file), `kernel`, `strategy` (`optimized|brute`), and optional `noise/add_noise` controls.
- **Response**: JSON containing runtime, PSNR, and base64 previews for original/noisy/denoised frames.
- **Static UI**: Not bundled in this cleaned release; supply your own frontend or call the API directly.

## Next Steps
- Populate `data/samples/` with test images (e.g., grayscale PNGs).
- Use the benchmarking script to capture runtime charts for the final documentation.
- Tweak algorithm parameters (window size, histogram bins) to target the desired accuracy/performance point.
