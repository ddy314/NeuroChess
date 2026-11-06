# Native MCTS prototype

Search is C++17: bitboard legal move generation, make/unmake, history-aware draw
claims, policy/value input encoding, PUCT, progressive widening, conservative
root pruning, exact mate proofs, virtual-loss batch selection, backup, and
subtree adoption/compaction. Python only supplies batch neural inference and
match orchestration. There are no Python callbacks inside native traversal.

Rules use the MIT-licensed [Disservin chess-library](https://github.com/Disservin/chess-library),
vendored at the exact commit in `neurochess/native/vendor/SOURCE.txt`. No new
Python environment or installed package is required.

## Build and use

From the repository directory, using the existing system compiler:

```sh
python scripts/build_native_mcts.py
```

`-O3 -march=native` builds for this machine; rebuild on a different CPU.
The compiled library is ignored by Git; source and license are included.
Build is explicit, with no automatic compiler invocation during play.
MCTS CLI now defaults to C++; append these options to an existing arena command:

```sh
--mode mcts --mcts-backend cpp --simulations 256 --leaf-batch-size 16 --root-keep 12
```

`--mcts-backend python` retains the original implementation for comparison.
The existing CUDA launcher remains `scripts/local_arena_existing_env.py`.
A single independent game can now produce a batch of leaves, rather than
requiring many simultaneously active games to use the GPU efficiently.

## Reuse, pruning, and memory

Each game slot retains its tree and neural evaluations between calls. The
initial FEN **and entire move-history prefix** must match before reuse; a FEN
match alone is insufficient because repetitions and model inputs depend on
history. One or several played plies can be followed into a previously searched
subtree. Missing descendants or incompatible histories reset that slot safely.
Stable slot order improves reuse; reordered games may lose reuse but cannot use
incompatible history. New moves advance the native root board by only the new
suffix. Siblings and old roots are discarded by compacting the retained tree.
The prototype uses one inference thread; it is not safe to share one player
instance between simultaneous callers.

Visits, priors, backed-up values, exact outcomes, and expanded descendants are
retained. Each move receives the full **additional** simulation budget; old
visits do not consume it. Root pruning restarts for each new search budget.
Nodes and edge blocks live in contiguous arenas; a scratch board uses
make/unmake, without copying full history at every leaf. Preallocated NumPy
buffers carry encoded inputs across the ABI, and policy/value return in one GPU
to CPU transfer; unused Q predictions are not copied back.

Virtual loss is removed for the whole batch before applying real values.
Pending leaves cannot be selected twice. Exact solver proofs still require all
legal children for a draw/loss; pruned unknown moves cannot establish a loss.
Pruning retains checks/captures/promotions, top priors, visit leaders and proven
wins. Full history is preserved for threefold claims (including intended next
moves), fifty-move claims, and repetition input features. Raw en-passant squares
for model inputs are tracked separately from legally capturable en-passant
squares used for repetition hashes. This prototype supports standard chess,
not variants, Chess960, or synthetic null-move histories.

Batching changes traversal order and thus may change chosen moves at a fixed
simulation count. Speed measurements do not establish unchanged playing
strength. The Python inference bridge remains; inference has not been ported
to LibTorch or TensorRT.

## Validation and measurements

```sh
.venv/bin/python -m pytest tests/test_native_mcts.py -q
python scripts/build_native_mcts.py --debug
.venv/bin/python -m pytest tests/test_native_mcts.py tests/test_mcts.py tests/test_oneply.py tests/test_local_arena.py tests/test_core.py -q
python scripts/build_native_mcts.py
python scripts/benchmark_mcts.py \
  --checkpoint /path/to/compatible-human-policy-checkpoint.pt \
  --simulations 128 --repeats 3 \
  --out output/native-mcts-20261030/benchmark.json
```

The historical step-8400 checkpoint used in this benchmark was removed during
the 2026-11-01 cleanup; the original benchmark results remain under
`output/native-mcts-20261030/`. A replacement checkpoint changes the benchmark.

Debug build enables undefined-behavior checks. Differential tests compare legal
action IDs, all model input fields, and exact terminal outcomes against
python-chess over random game histories and rule boundary fixtures. Tactical
checks cover mate with a tiny prior and avoiding Fool's mate; search checks
cover virtual-loss budgeting, pruning, input immutability, batch slots,
new-move budgets and incompatible-history rejection. Benchmark JSON separately
records neural-free search timings, actual checkpoint/GPU timings and a 12-ply
legal self-play trace demonstrating subtree reuse. Model load time is excluded;
all methods receive the same simulation budget and warmup. This is a local
latency benchmark, not a strength or Elo evaluation.

The direct C++ invariants and reference perft counts can also be checked without
Python inference:

```sh
g++ -std=c++17 -O1 -g -fsanitize=undefined -fno-sanitize-recover=all \
  tests/native_mcts_core.cpp -o /tmp/neurochess-native-mcts-tests
/tmp/neurochess-native-mcts-tests
```

## Local two-second baseline

The measured local profile is `configs/puct-2s-local.json` (RTX4060Laptop,
original frozen multihead checkpoint). It enables `time_limit=2.0`, batch64,
a32768-entry effective-input inference cache, subtree reuse, and
`pending_loss_guard=True`. See the [measurements and quality controls](../reports/PUCT_2S_BASELINE_20261031.md).

```python
options = json.loads(Path('configs/puct-2s-local.json').read_text())
player = NativeMCTSPlayer(network, **options)
move = player.choose(board)
```

The generic MCTS CLI accepts `--search-config`, `--search-seconds`, and
`--inference-cache-size`; JSON config options override individual flags.
Time control currently requires one game slot. It stops between completed
batches, reserving observed batch latency; it is an approximate wall-time
budget, not a hard real-time guarantee. The simulation count is a safety cap
in this mode. Root pruning uses elapsed-time fraction rather than a fraction
of that cap. Fixed-count defaults retain their previous pruning behavior.

The inference cache never merges rule states, proofs, visits, or search edges.
It keys exact effective model inputs plus ordered legal action IDs and also
deduplicates a batch. Call `clear_cache()` after changing model weights or head
fusion settings; it discards both cached outputs and the retained search tree.

The batch safety guard classifies viable alternatives before masking pending
leaves. If all viable leaves await inference, it flushes the partial batch
rather than repeatedly backing up a known losing child. This avoids inflating
simulation counts while starving network evaluation. It is explicit in the new
profile; `pending_loss_guard=False` preserves the original comparison behavior.
