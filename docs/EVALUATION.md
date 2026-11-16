# Evaluation

## Playing strength vs. Stockfish 18

NeuroChess (INT8 TensorRT engine) was tested against Stockfish 18 (`Threads=1`, 64 MB transposition table, `Skill Level 20`) using color-paired openings (each opening played twice with colors reversed). NeuroChess used batch size 16, `--q-prior-beta 2`, and a 32,768-entry evaluation cache.

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

At lower search budgets, NeuroChess scored 66.7% / 43.8% / 12.5% against 1,000 / 2,000 / 4,000 Stockfish nodes at 256 simulations, and 84.4% / 50.0% / 34.4% / 12.5% against 2,000 / 4,000 / 8,000 / 16,000 Stockfish nodes at 2,048 simulations.

## Move latency

A fixed-work benchmark compared the native C++ engine against the Python search runtime across 12 positions using TensorRT and 8,192 simulations per position on an NVIDIA RTX 4060 Laptop GPU:

| Runtime | Median | P95 |
|---|---:|---:|
| Native C++ | 1.65 s | 2.19 s |
| Python | 1.90 s | 2.40 s |

The native C++ runtime was 1.15x faster at median move time, and both runtimes selected the same move on 11 of 12 positions.
