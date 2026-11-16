# Evaluation

## Playing strength vs. Stockfish 18

NeuroChess (`neurochess-13750.pt`, INT8 TensorRT engine) was tested against Stockfish 18 (`Threads=1`, `Hash=64 MB`, `Skill Level 20`) using color-paired openings (each opening played twice with colors reversed). NeuroChess used batch size 16, `--q-prior-beta 2`, and a 32,768-entry evaluation cache.

### Scaling with simulation budget

| NeuroChess simulations / move | Games | Equal-score Stockfish 18 budget | 90% paired bootstrap interval |
|---:|---:|---:|---:|
| 256 | 64 | 1,602 nodes / move | 1,210–1,996 nodes |
| 2,048 | 64 | 4,685 nodes / move | 3,811–5,860 nodes |
| 8,192 | 128 | 7,981 nodes / move | 5,782–10,361 nodes |

### Match scores at 8,192 simulations (128 games)

| Stockfish 18 budget | Opening pairs | NeuroChess score |
|---:|---:|---:|
| 4,000 nodes / move | 20 | 61.3% |
| 8,000 nodes / move | 16 | 51.6% |
| 16,000 nodes / move | 12 | 43.8% |
| 32,000 nodes / move | 16 | 20.3% |

Across the lower simulation budgets, NeuroChess scored 66.7% / 43.8% / 12.5% against 1k / 2k / 4k Stockfish nodes at 256 simulations, and 84.4% / 50.0% / 34.4% / 12.5% against 2k / 4k / 8k / 16k Stockfish nodes at 2,048 simulations.

### Search vs. raw network policy

Tree search drives a large jump in playing strength over single-pass policy output. On an earlier engine-supervised checkpoint tested against 1,000-node Stockfish 18 (64 games each), raw network moves scored 1W / 9D / 54L (8.6%), whereas adding 512 simulations of PUCT search raised the score to 29W / 14D / 21L (56.3%).

## Move latency

A fixed-work benchmark compared the native C++ engine against the Python search runtime across 12 positions using TensorRT and 8,192 simulations per position on an NVIDIA RTX 4060 Laptop GPU:

| Runtime | Median | P95 |
|---|---:|---:|
| Native C++ | 1.65 s | 2.19 s |
| Python | 1.90 s | 2.40 s |

The native C++ runtime was 1.15x faster at median move time, and both runtimes selected the same move on 11 of 12 positions.
