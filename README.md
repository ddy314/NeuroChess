# NeuroChess

NeuroChess is a chess engine built around a 155.7M-parameter Transformer and PUCT tree search. Instead of a convolutional board tower, it encodes the 64 squares as tokens, scores all legal moves in a single forward pass with policy, action-value, and win/draw/loss heads, and searches through a native C++/CUDA engine powered by INT8 TensorRT inference.

[Download the model](https://github.com/ddy314/NeuroChess/releases/tag/model-v1) · [Model and engine](docs/MODEL.md) · [Evaluation](docs/EVALUATION.md) · [Training](docs/TRAINING.md)

## How it works

| Component | Design |
|---|---|
| Backbone | 16-layer pre-norm Transformer (896 channels, 14 heads, 155.7M parameters) |
| Input | Side-to-move square tokens (64 squares), position metadata, and legal move IDs |
| Policy head | Scores every legal move simultaneously from source, destination, and promotion embeddings |
| Action-value head | Cross-attends each legal move to the board to predict a 128-bin value distribution over `[-1, 1]` |
| Value head | Predicts win, draw, and loss probabilities for the side to move |
| Search | Batched C++ PUCT with Q-guided priors, LRU evaluation cache, and subtree reuse |
| Runtime | C++17, CUDA Graphs, TensorRT INT8, and UCI |

During search, the C++ engine folds the per-move action-value predictions into the search priors (`--q-prior-beta`), batches uncached leaf positions with virtual loss, executes INT8 TensorRT inference via CUDA Graphs, and reuses the surviving search subtree across consecutive moves.

## Results

In paired-opening matches against Stockfish 18 (1 thread, 64 MB hash, full strength), NeuroChess scales consistently with search budget:

| NeuroChess simulations / move | Games | Equal-score Stockfish 18 budget | 90% paired bootstrap interval |
|---:|---:|---:|---:|
| 256 | 64 | 1,602 nodes / move | 1,210–1,996 nodes |
| 2,048 | 64 | 4,685 nodes / move | 3,811–5,860 nodes |
| 8,192 | 128 | 7,981 nodes / move | 5,782–10,361 nodes |

On an NVIDIA RTX 4060 Laptop GPU at 8,192 simulations per move, the native C++/TensorRT engine runs in **1.65 s median** (**2.19 s P95**) per move. See [Evaluation](docs/EVALUATION.md) for the per-level score breakdown and latency comparison.

## Quick start

### Evaluate a position

Requires Python 3.11 or later.

```bash
git clone https://github.com/ddy314/NeuroChess.git
cd NeuroChess
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
mkdir -p models
curl -L -o models/neurochess-13750.pt \
  https://github.com/ddy314/NeuroChess/releases/download/model-v1/neurochess-13750.pt
python -m neurochess.predict --model models/neurochess-13750.pt
```

This evaluates the starting position with the raw network (no tree search) and prints the side-to-move WDL probabilities along with each top move's prior probability and expected action value `Q`. Pass `--fen '<FEN>'` for another position or `--device cpu` to run on CPU.

### Play over UCI

The native engine requires an NVIDIA GPU, CUDA Toolkit, TensorRT 10+, CMake 3.24+, and a C++17 compiler. Install the `tensorrt` Python package matching your local TensorRT SDK, then export the TensorRT plan and build the engine on the target machine:

```bash
pip install -e '.[export]'
python -m neurochess.export --model models/neurochess-13750.pt --out models/
cmake -S engine -B engine/build -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build engine/build -j
```

Launch the UCI engine:

```bash
./engine/build/neurochess --engine models/neurochess-13750-int8.plan \
  --uci --history-free --batch 16 --q-prior-beta 2 --cache 32768 \
  --time -1 --simulations 8192
```

Register this command in any UCI-compatible chess GUI. See [Model and engine](docs/MODEL.md) for architecture details, quantization, and search options.

## Repository

| Directory | Contents |
|---|---|
| [`neurochess/`](neurochess/) | Transformer model, board encoding, Python evaluation, and TensorRT INT8 export |
| [`engine/`](engine/) | Native C++/CUDA UCI engine, PUCT search, and TensorRT runtime |
| [`docs/`](docs/) | Architecture, evaluation, and training documentation |
| [`tests/`](tests/) | Model unit tests and UCI integration tests |

Run `pytest -q` for the model tests. UCI integration tests run automatically once `engine/build/neurochess` and `models/neurochess-13750-int8.plan` exist (override paths with `NEUROCHESS_ENGINE` and `NEUROCHESS_PLAN`).

Move generation uses [Disservin/chess-library](https://github.com/Disservin/chess-library) under the MIT license.
