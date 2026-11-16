# NeuroChess

**NeuroChess** is a 155M-parameter Transformer chess engine paired with a native C++17 / CUDA PUCT search and TensorRT INT8 inference.

A single encoder pass over the 64 board squares produces legal-move policy logits, per-move action-value histograms $Q(s, a) \in [-1, 1]$, and a win / draw / loss evaluation. During search, $Q$ reshapes the PUCT prior via a KL-regularized update $\pi'(a) \propto \pi(a)\exp(\beta\, Q(s, a))$ while a C++ search thread pool drives an asynchronous CUDA-Graph TensorRT pipeline with an out-of-order neural evaluation cache.

---

## Highlights

- **Square-token Transformer (`155,691,271` parameters)** — 16 layers, $d_{\text{model}} = 896$, 14 heads, factorized source / destination / promotion move queries, and a 128-bin Distributional Action-Value ($Q$) head attending to the board encoding.
- **Joint P / Q / V Checkpoint (`neurochess-13750`)** — jointly trained with Muon + AdamW over 56.3M sample presentations (`step 13,750`, effective batch 4,096) across deep Stockfish multi-PV evaluations, human game replay, ChessBench, and Syzygy tablebases.
- **Zero-Python UCI Runtime** — standalone C++17 / CUDA executable (`engine/build/neurochess`) speaking standard UCI, with pending-loss virtual visits, an LRU neural cache, SmoothQuant INT8 Transformer matmuls, and per-batch CUDA Graph capture.

---

## Repository Layout

```text
.
├── engine/                         # Standalone C++17 / CUDA UCI engine
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp                # UCI loop, cache, and batch scheduler
│   │   ├── mcts.hpp                # Native PUCT search core
│   │   ├── chess_rules.hpp         # Move generation and board encoding
│   │   ├── trt_inference.hpp
│   │   └── trt_inference.cu        # TensorRT runtime + CUDA Graph execution
│   └── third_party/chess-library/  # Vendored Disservin/chess-library (MIT)
├── neurochess/                     # Minimal Python package (model & TensorRT export)
│   ├── model.py                    # NeuroChessModel definition & checkpoint loader
│   ├── encoding.py                 # Side-to-move board & move encoding
│   ├── predict.py                  # Raw network evaluation CLI (no search)
│   ├── export.py                   # SmoothQuant INT8 ONNX -> TensorRT builder
│   └── calibration.fen             # 1,024 calibration positions for INT8 export
├── tests/                          # Pytest suite (model unit tests + UCI integration)
└── pyproject.toml
```

---

## Model & Playing Strength

The primary released model is **`neurochess-13750`** (`joint-muon-pqv-13750-v1`):

| Property | Value |
| :--- | :--- |
| Architecture | `action_value_histogram_v2` (16L, $d = 896$, 14H, $d_{\text{head}} = 256$, $d_Q = 128$, 128 bins) |
| Parameters | `155,691,271` |
| Training step | `13,750` (`56,320,000` presentations, effective batch `4,096`) |
| Weights SHA-256 | `910e517774770754500ff3ad6b66380d0bf3d36f5e1f85f7b2451407b95c18a7` |

### Fixed-Node Match Results vs. Stockfish 18

Evaluated with the native INT8 TensorRT engine (`--batch 16 --q-prior-beta 2 --cache 32768 --history-free`) across color-paired opening suites against Stockfish 18 at fixed node budgets (not a calibrated human Elo rating):

| PUCT Simulations / Move | Equivalent SF18 Nodes / Move | 90% Paired Resampling Interval | Games |
| ---: | ---: | ---: | ---: |
| 256 | ~1,600 | 1,210 – 1,996 | 64 |
| 2,048 | ~4,700 | 3,811 – 5,860 | 64 |
| 8,192 | ~8,000 | 5,782 – 10,361 | 128 |

At 8,192 simulations per move on an RTX 4060 Laptop GPU, the native INT8 runtime achieves a median move time of **1.65 s** (P95 **2.19 s**), a **4.6×** end-to-end speedup over eager BF16 PyTorch inference.

> **Search baseline note:** For search optimization work, the locked two-second PUCT quality floor remains documented in `configs/search-baseline-lock.json` and `docs/SEARCH_BASELINE.md` when present in a research checkout. Speed and playing quality are always evaluated and reported separately.

---

## Quick Start

### 1. Python Package (Network Inspection & TensorRT Export)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
```

Evaluate a position directly with the PyTorch checkpoint (no search):

```bash
python -m neurochess.predict --model models/neurochess-13750.pt \
  --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
```

```text
WDL (side to move): 0.057 / 0.942 / 0.001
move       prior       Q
Nc3        0.247  +0.025
d4         0.245  +0.033
Bc4        0.233  +0.026
Bb5        0.197  +0.018
```

Or load the model in Python:

```python
import chess
from neurochess import load_model
from neurochess.predict import evaluate

model = load_model("models/neurochess-13750.pt", device="cuda")
info = evaluate(model, chess.Board())
```

### 2. Build the TensorRT INT8 Engine

TensorRT `.plan` files are specific to the target GPU and TensorRT version. To export `neurochess-13750.pt` into a SmoothQuant INT8 engine on your machine (requires `onnx`, `ml_dtypes`, and the `tensorrt` Python wheel):

```bash
python -m neurochess.export --model models/neurochess-13750.pt --out models/
```

This calibrates activation maxima over the 1,024 positions in `neurochess/calibration.fen`, quantizes the 64 Transformer matmuls to INT8 while keeping the heads in BF16/FP32, and writes `models/neurochess-13750-int8.plan`.

### 3. Build & Run the Native UCI Engine

Requires a C++17 compiler, CUDA Toolkit, and TensorRT 10+ / 11 headers and libraries:

```bash
cmake -S engine -B engine/build -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build engine/build -j
```

Launch the engine in UCI mode (compatible with Cutechess, En Croissant, Nibbler, Arena, etc.):

```bash
./engine/build/neurochess \
  --engine models/neurochess-13750-int8.plan \
  --uci --history-free --batch 16 --q-prior-beta 2 --cache 32768 \
  --time -1 --simulations 8192
```

Or query a single position from the command line:

```bash
./engine/build/neurochess \
  --engine models/neurochess-13750-int8.plan \
  --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" \
  --history-free --batch 16 --q-prior-beta 2 --time -1 --simulations 2048
```

---

## Testing

```bash
pytest -q
```

If `engine/build/neurochess` and a built `.plan` file are present, set `NEUROCHESS_PLAN=models/neurochess-13750-int8.plan` to include the end-to-end UCI integration tests.

---

## Acknowledgments

- Move generation in the native engine uses [Disservin/chess-library](https://github.com/Disservin/chess-library) (MIT License, vendored in `engine/third_party/chess-library/`).
