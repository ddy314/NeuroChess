# Model and engine

## Board and move representation

NeuroChess encodes every position from the side-to-move's perspective: when Black is to move, ranks are flipped (`square ^ 56`) so the model sees White and Black symmetrically, while files remain unflipped to keep kingside and queenside castling semantics consistent.

Each position is passed as three tensors (see [`encoding.py`](../neurochess/encoding.py)):

- **`tokens` `(64,)`**: Piece ID (`0` empty, `1–6` own P/N/B/R/Q/K, `7–12` opponent P/N/B/R/Q/K) for each square.
- **`meta` `(12,)`**: Castling rights (4), en-passant square and flag (2), halfmove clock, 2-fold and 3-fold repetition flags, check flag, side to move, and fullmove counter.
- **`moves` `(256,)`**: Legal moves encoded as `(from * 64 + to) * 5 + promotion` in the side-to-move frame, padded with `-1`.

## Network architecture

A 16-layer pre-norm Transformer (`896` channels, `14` attention heads, `GELU` feed-forward blocks; `155,691,271` total parameters) processes the 64 square tokens conditioned on the metadata vector, then evaluates all legal moves in a single pass ([`model.py`](../neurochess/model.py)):

| Head | How it works | Output |
|---|---|---|
| **Policy** | Combines source-square, destination-square, scaled elementwise interaction (`a * b / sqrt(d)`), and promotion embeddings into a `256`-dim move representation | Logits over up to 256 legal moves |
| **Action value (`Q`)** | Builds a `128`-dim query per legal move (with global board context), cross-attends to the 64 square tokens, and predicts a `128`-bin histogram over `[-1, 1]` | Expected scalar `Q` in `[-1, 1]` per move |
| **Position value (WDL)** | Projects the mean-pooled 64 square representations through a 2-layer MLP | Win, draw, and loss logits |

Predicting a 128-bin distribution rather than a direct scalar for `Q` stabilizes learning on sharp tactical swings and lets the search combine move-level value estimates directly with policy priors.

## SmoothQuant and TensorRT INT8 export

The exporter ([`export.py`](../neurochess/export.py)) converts the PyTorch checkpoint into a hardware-tuned TensorRT `.plan` file:

1. **Activation calibration**: Runs 1,024 diverse positions from `neurochess/calibration.fen` to measure per-channel activation maxima for all 64 Transformer linear layers (`qkv`, `out`, and the two FFN projections across 16 layers).
2. **SmoothQuant (`alpha = 0.5`)**: Scales weights and activations per input channel to balance quantization range, then inserts INT8 `QuantizeLinear` / `DequantizeLinear` pairs around the 64 Transformer matrix multiplications in the exported ONNX graph.
3. **Mixed precision**: Embeddings, LayerNorms, attention softmax, and the policy/Q/WDL heads stay in BF16/FP32.

Because TensorRT plans are compiled for a specific GPU architecture and TensorRT version, export the `.plan` file on the machine where the engine will run.

## Native search and UCI runtime

The native engine ([`main.cpp`](../engine/src/main.cpp), [`mcts.hpp`](../engine/src/mcts.hpp), [`trt_inference.cu`](../engine/src/trt_inference.cu)) runs search and inference entirely in C++/CUDA:

- **Q-guided priors (`--q-prior-beta`)**: A fused CUDA kernel shifts each legal move's policy logit by `beta * (Q - E_p[Q])`, boosting tactically sound moves before search even expands them. Inside the tree, `Q` also provides two virtual observations on unvisited edges before falling back to WDL first-play urgency.
- **History-free mode (`--history-free`)**: Masks out the halfmove clock, repetition flags, and fullmove number so evaluations depend purely on board state and maximize cache hits across transpositions.
- **Batched PUCT with virtual loss**: Collects up to `--batch` distinct uncached leaves per inference round using virtual loss, deduplicates repeated positions within the batch, and checks an LRU evaluation cache (default 32,768 entries).
- **CUDA Graphs**: Pre-captures host-to-device transfers, TensorRT inference, and head-fusion kernels for every batch size `1..batch` to eliminate kernel-launch overhead.
- **Tactical proofs & progressive widening**: Detects mates-in-1 on node expansion, propagates exact win/loss proofs up the tree, widens exploration progressively (`6 + 1.5 * sqrt(visits)` moves plus forcing checks, captures, and promotions), prunes weak root moves after 75% of the budget, and reuses the retained subtree across moves.

### CLI options

| Flag | Default | Description |
|---|---:|---|
| `--engine <path>` | required | Path to the compiled TensorRT `.plan` file |
| `--uci` | off | Run in UCI loop mode (otherwise evaluates `--fen`) |
| `--simulations <N>` | `16384` | Simulation budget per move |
| `--time <sec>` | `0.5` | Time limit per move in seconds (`-1` for simulation-only search) |
| `--batch <N>` | `8` | Inference batch size (`1`–`16`) |
| `--q-prior-beta <val>` | off | Weight for mixing centered action-value `Q` into policy priors (`p + beta * (Q - E_p[Q])`) |
| `--cache <N>` | `32768` | LRU evaluation cache capacity in positions (`0` disables) |
| `--history-free` | off | Masks halfmove clock, repetition flags, and fullmove number |
| `--no-graphs` | off | Disables CUDA Graph capture |

See [Quick start](../README.md#quick-start) for build and launch commands.
