# Sparse-query best-first minimax experiment

This experiment separates inexpensive CPU expansion from expensive Transformer
queries. It is an alternative experimental player, not the default MCTS player.
Strength parity with native PUCT has **not** been established.

The follow-up [tactical diagnosis and controlled comparisons](../reports/SEARCH_DIAGNOSIS_20261031.md)
separate capacity, search-distribution error, target saturation, and search/query
strategy. The original defaults remain available for reproducing the baseline.
The following explicit experimental settings align root exploration, query
priorities, and final selection with the same policy-regularized objective:

```python
player = BestFirstPlayer(weights, network, expansions=512, neural_budget=32,
                         root_policy_weight=.1, align_root_objective=True)
```

The arena accepts `--root-policy-weight .1 --align-root-objective`.
`--methods best-first,mcts-small,mcts` includes a matched low-query PUCT control
(31 additional simulations, leaf batch 4) and the original 512-simulation control.
`--resume` continues completed method blocks only with an identical plan and
input fingerprints; it does not resume an unfinished individual game.

`verify_final=True` reserves up to eight positions **inside** the existing query
budget for checking the final root candidate. `final_candidate_checked` means
directly evaluated (possibly cached) or exactly solved, not correctly evaluated.
If no child query is affordable, the search uses a policy fallback and retains
exact-outcome priority. This remains off by default: it closes a query-coverage
gap, but some real replays lose quality because the evaluator itself is wrong.

`trace()` exposes the full partial tree and correction events for diagnostics.
Optional ablations include `anchor_weight`, `q_anchor_weight`, `anchor_decay`,
and `correction_scheduler='frontier'`. They were tested rather than presumed to
improve strength; removing anchors or simply querying deeper was not reliable.

## Implementation

`neurochess/cheap.py` defines a 107,073-parameter value network with the default
128-wide shared piece-square accumulator, two perspectives, a 32-unit hidden
layer, and a material residual. Its target is the side-to-move W-L proxy of an
engine-searched score. The model is an NNUE-style float prototype; it does not
use Stockfish's architecture, quantization, or weights. Raw default weights are
about 419 KiB. Training can use local CUDA; inference runs in C++ on CPU.

`neurochess/native/best_first.cpp` maintains a partial game tree with negamax
backup and selective best-first expansion. Every expansion materializes **all**
legal children. Every sixteenth expansion uses least-explored selection to
reduce starvation. This is a heuristic finite-budget search, not a completeness
claim or a reproduction of Athénan.

The evaluator keeps accumulators in both White and Black frames. Make/unmake
traversal updates them using changed piece-square features; special moves are
handled by before/after piece bitboards, including castling, en passant, and all
promotions. Metadata is history-free to match the engine-label dataset. Actual
terminal decisions always use the full legal history.

Optional bounded CPU quiescence covers captures, promotions, and all legal
check evasions. Quiescence is disabled by default: the depth-2 development
ablation added substantial CPU time without a consistent quality benefit. Set
`quiescence_depth=2` to enable it, with at most 100,000 extra quiescence nodes per
move. Depth/node limits and quiescence estimates never establish
exact game-theoretic outcomes.

`BestFirstPlayer` gives the Transformer a separate **hard per-move position
budget**, including the root. With `neural_budget=32`, at most 32 positions are
requested; cache hits can reduce actual forwards further. The CPU expansion
budget remains independent. Root evaluation happens first. Later correction
rounds are spread through the expansion budget, and query candidates are ranked
by cumulative minimax gaps. High-level candidate nodes are checked before their
descendants. These priorities are heuristics, not calibrated confidence bounds.

Corrections remain as soft anchors after a node is expanded, rather than being
discarded when cheap child scores replace its evaluation. For an expanded,
unresolved node:

```
value = (1 - anchor_weight) * max(-child.value) + anchor_weight * anchor
```

Direct Transformer WDL anchors use weight .75; policy-supported candidate-Q
anchors use .25. These are fixed prototype choices, not tuned or theoretically
justified probabilities. Consequently, this implementation is **regularized
minimax**, not pure minimax over one homogeneous frontier evaluator. Exact
mate/draw/loss outcomes override anchors. A loss or draw proof still requires
all legal children; an estimated value close to +/-1 is never a proof.

The Python bridge supports the existing `MultiheadPlayer` and the simpler
`evaluate_encoded()` interface. A bounded LRU caches outputs by effective model
inputs **and ordered legal action IDs**. History-free models mask only the
metadata fields actually masked by `ChessModel`; rule states remain separate.
Call `clear_cache()` after changing model weights or inference configuration.
Tree statistics are not merged across histories. The default rebuilds its tree
for each move. Experimental `tree_reuse=True` retains the played subtree only
when the initial FEN and complete move-history prefix agree. It compacts node
indices, rebases depth, reopens unresolved depth cutoffs, and preserves exact
proofs and neural observations. The expansion/query caps apply anew each move;
retained work does not consume them. `clear_cache()` also discards this tree.
`choose_many()` searches games sequentially. Do not share a player across threads.

## Causal search controls

All new controls are opt-in; `backup_mode='legacy'`, `cheap_evaluator='student'`,
`tree_reuse=False`, and `root_coverage=0` preserve previous search behavior.
`NativeMCTSPlayer(tree_reuse=False)` disables only subtree retention for the
PUCT control and keeps cumulative statistics and the simulation budget.

`backup_mode='visit-mean'` averages current child estimates using CPU expansion
visits before applying the original anchor. This is a diagnostic operator, not
MCTS sample backup. `evidence-minimax` uses neural-evaluated descendants or
policy-supported Q as the backed frontier, falling back to cheap values when
no such evidence exists. Its final ranking prefers these evaluated alternatives.
Neural evidence can itself be wrong; this mode is not a confidence bound.

`backup_mode='neural-mean'` averages signed direct neural observations in each
subtree, with supported parent Q contributing two pseudo-observations. Each
directly queried tree node contributes once; repeated root queries do not add
another sample. These observations are correlated, and counts are not an
effective statistical sample size. Neither alternative changes proof rules.

`root_coverage=8` first directly calibrates the eight highest-policy unresolved
root moves before resuming gap-based correction. It uses the original position
cap, rather than adding queries. This tests premature concentration of the
correction budget independently of value backup. `cheap_evaluator='material'`
and `'neutral'` disable learned student evaluation throughout cheap expansion
and CPU value probing; exact mate search is unchanged. A CP-head export can be
generated with `CheapModel.export(path, head='cp')` for a head-choice ablation;
it returns `tanh(predicted_cp/600)`, rather than a calibrated WDL proxy.

Traces expose raw cheap/direct/backed values, supported Q, neural observation
counts, and retained-node statistics. The controlled results and limitations
are in [the strength diagnosis](../reports/STRENGTH_CAUSES_20261031.md).

## Local training

The initial student used the existing 851,213-position engine-evaluation dataset.
CUDA continuation uses `data/evals-v2-5m` (5,000,000 train / 51,086 validation
positions). Training samples the existing searched-value labels; it does not
require running the large Transformer.

The selected, actually trained CPU weights are available at
`output/cheap-minimax-20261031/student-selected/best.bin`, with the PyTorch
checkpoint and selection/validation record in the same directory. The requested
additional 100,000 GPU training steps were completed as 75,000 + 25,000 steps
after a SIGTERM interruption; recovery restarted the optimizer from saved weights.
See `reports/BEST_FIRST_MINIMAX_20261031.md` for measured limitations and results.

Use a Python environment with CUDA-enabled PyTorch and the project dependencies installed.

```sh
python scripts/train_cheap.py \
  --data data/evals-v2-5m --out output/student-new \
  --device cuda --batch-size 4096 --steps 100000 --threads 2 \
  --lr 0.0002 --cosine --validate-every 2500
```

Use `--init PATH/best.pt` for weights-only initialization of a new training stage;
the optimizer and schedule restart. This is not exact optimizer/cursor resume.
Use a new output directory for each run. `best.pt`/`best.bin` track lowest
validation MSE; `last.*` track the latest evaluation checkpoint. `metrics.jsonl`,
`run.json`, and `status.json` preserve metrics, data manifest, seed, and arguments.
The standard validation pass caps at the first 20,000 validation rows. The split
is by position hash; it is not game- or source-independent and MSE is not Elo.

Fresh local teacher labels can be generated with existing tools:

```sh
.venv/bin/python -m neurochess.sample_fens \
  --pgn data/source/lichess-2013-01.pgn.zst \
  --out output/student-teacher.fen --positions 2048 --min-elo 2000 --seed 20261031
.venv/bin/python -m neurochess.distill \
  --fens output/student-teacher.fen \
  --stockfish local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2 \
  --out data/student-teacher-new --workers 4 --threads 1 --nodes 20000 --multipv 1
```

This was actually run for 2,048 positions in `data/cheap-teacher-20261031`.
Those fresh labels were reserved for development validation, not used in the
initial student runs. The overlap audit excluded 97 exact keys present in the
full v2 dataset, leaving 1,951 positions; related games may still overlap.

## Build and play

```sh
python scripts/build_best_first.py
```

Minimal CPU-only usage, with no large-model inference:

```python
import chess
from neurochess.best_first import BestFirstPlayer
player = BestFirstPlayer("output/student-new/best.bin", expansions=512)
move = player.choose(chess.Board())
print(move, player.search_stats)
player.close()
```

Pass a loaded `MultiheadPlayer` as `network=` to enable sparse corrections.
Native compilation is explicit, and stale-source checks require rebuilding after
changes. Native rules/encoding live in `chess_rules.hpp`, shared with the existing
PUCT engine, whose behavior is covered by the existing differential tests.

Local paired arena, with PGNs, exact input fingerprints, truncation counts, and
actual large-model position counts:

```sh
python scripts/best_first_arena.py \
  --cheap output/student-new/best.bin \
  --checkpoint output/eval-multihead-20261031/model/best.pt \
  --stockfish local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2 \
  --openings output/eval-multihead-20261031/openings.fen \
  --out output/best-first-arena-new --pairs 4 \
  --expansions 256 --neural-budget 32 --methods best-first,mcts
```

Use `--methods cheap --device cpu` without `--checkpoint` for entirely CPU play.
The arena uses the existing claimable-draw adjudication. Reaching a ply/time cap
produces a truncation, not a draw. Opening roots were used during development;
these games are not a final independent strength test.

## Reproducible verification

```sh
python scripts/build_best_first.py --debug
.venv/bin/python -m pytest tests/test_best_first.py tests/test_native_mcts.py tests/test_mcts.py -q
g++ -std=c++17 -O1 -g -fsanitize=undefined -fno-sanitize-recover=all \
  tests/native_best_first_core.cpp -o /tmp/neurochess-best-first-core
/tmp/neurochess-best-first-core output/student-new/best.bin
python scripts/build_best_first.py
.venv/bin/python -m pytest -q
```

Tests cover native/PyTorch evaluation agreement, incremental updates across
random legal histories and special moves, minimax signs, persistent anchors,
exact-proof boundaries, hard query budgets, correction effects, output caching,
terminal claims, mate refutations, and node/depth caps.

For cold-position measurements and Stockfish candidate-union regret:

```sh
python scripts/benchmark_best_first.py \
  --cheap output/student-new/best.bin \
  --checkpoint output/eval-multihead-20261031/model/best.pt \
  --stockfish local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2 \
  --openings output/eval-multihead-20261031/openings.fen \
  --out output/best-first-benchmark-new --budgets 0,8,16,32 \
  --expansions 512 --simulations 512 --quiescence-depth 0
```

Counts distinguish queried positions from forward batches. Teacher regret is
relative to the union of proposed moves searched coherently by Stockfish; it is
not regret against the exact optimal move. CPU/GPU contention affects timings.
Compare equal wall time as well as equal large-model position counts before
drawing conclusions about usefulness.

## Current limitations

Initial eight-position measurements reduced expensive evaluation counts by
roughly 16x at budget 32, but did **not** preserve PUCT's move quality. Persistent
anchors and CPU quiescence were added as separate development iterations.
Quiescence increases CPU work substantially and still does not eliminate the
student's tactical errors. The experiments are intentionally preserved under
`output/cheap-minimax-20261031/benchmark-v{1,2,3}` rather than presenting speed
alone as success. See the accompanying experiment report for final long-training
and arena results.

Useful next work is search-frontier-specific distillation, stronger uncertainty
calibration/query scheduling, incremental leaf reuse, and evaluation on fresh
openings. Merely increasing cheap expansions can amplify evaluation errors.
