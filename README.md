# Accelerating Median Filtering for Image Noise Reduction

Course final project for CpE 411 (Data Structures and Algorithms) – Batangas State University TNEU. The system delivers a high-performance median filter for salt-and-pepper noise removal by combining three synergistic data structures: 2D arrays for pixel storage, a priority-queue pair (dual heap) for O(log k) median maintenance, and a tree-based array (Fenwick/segment tree) for fast histogram queries augmented with a dictionary cache. Benchmarks compare the optimized filter against a brute-force baseline to quantify speedups and fidelity.

## Core Requirements
- **Data Structures**: 2D Array, Priority Queue Pair (Min/Max heaps), Fenwick/Segment Tree, Dictionary cache.
- **Algorithms**:
  - Optimized sliding-window median maintenance using dual heaps.
  - Fenwick-tree range-query aggregation for histogram-aware filtering.
  - Baseline brute-force median filtering for comparison.
- **Language/Stack**: Python 3.11+, NumPy, Pillow, Matplotlib, scikit-image, Flask.

## Repository Layout
```
.
├── .flaskenv
├── app.py
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
│   ├── test_dual_heap.py
│   ├── test_filters.py
│   └── test_segment_tree.py
├── web/
│   ├── app.js
│   ├── index.html
│   ├── styles.css
│   └── assets/
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
   python -m dsa_median.benchmarks --image path/to/image.png --kernel 5 --noise 0.1
   ```
4. Launch the Flask API (serves SPA + JSON endpoint). The repo includes `.flaskenv` and a thin `app.py`, so a plain `flask run` works (python-dotenv is already listed as a dependency):
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # only if scripts are blocked
   .\.venv\Scripts\Activate.ps1
   flask run --no-reload
   ```
   This starts `src/dsa_median/webapp.py` on port 8000. Visit http://localhost:8000 to use the Polar Median Lab UI, or call `/api/denoise` directly from integration tests.
5. Capture benchmark figures or logs from the CLI run to document acceleration and fidelity.

## Filtering Backends
- `optimized_median_filter(..., backend="auto")` (Polar Median Lab UI default) now prefers the fast NumPy/vectorized backend so the optimized strategy outpaces the intentionally slow brute-force baseline. Pass a positive `auto_threshold` (pixel count) if you want small demo images to fall back to the dual-heap/Fenwick/dictionary pipeline automatically.
- `backend="heap"` forces the data-structure-rich implementation regardless of image size—useful when you need to highlight the four DS requirements explicitly.
- `backend="vectorized"` explicitly opts into the NumPy backend. For `backend="auto"` this is already the behavior unless you tweak `auto_threshold`.

## Web/API Notes
- **Static UI**: `web/` contains a lightweight SPA. When the Flask app runs, `/` serves `index.html` and `/static` exposes the remaining assets.
- **Endpoint**: `POST /api/denoise` accepts `image` (file), `kernel`, `strategy` (`optimized|brute`), and optional `noise/add_noise` controls.
- **Response**: JSON containing runtime, PSNR, and base64 previews for original/noisy/denoised frames.
