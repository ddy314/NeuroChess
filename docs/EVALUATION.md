# Evaluation

## Stockfish comparison

The released checkpoint was tested with a native INT8 TensorRT engine and Stockfish 18. NeuroChess used 8,192 PUCT simulations per move, batch size 16, action-value prior beta 2 and a 32,768-entry cache. Stockfish used one thread, a 64 MB hash and full skill. Games used colour-paired openings.

| NeuroChess simulations | Games | Estimated equal-score Stockfish nodes per move | 90% paired resampling interval |
|---:|---:|---:|---:|
| 8,192 | 128 | 7,981 | 5,782–10,361 |

The estimate fits results across several Stockfish node budgets. Its interval is conditional on the sampled openings and fitted model. The estimate describes this protocol; no human Elo calibration has been performed. The [combined analysis](evidence/stockfish-high-budget-analysis.json) contains the fit and paired results. The [earlier refinement audit](evidence/stockfish-refinement-audit.json) records the first set of games. The model SHA in the experiment plan matches the release weights.

## Runtime

A 12-position fixed-work comparison on an RTX 4060 Laptop GPU used 8,192 simulations per position. The native and Python paths both used TensorRT. Median times were 1.648 s and 1.901 s, respectively; P95 times were 2.188 s and 2.400 s. The native path was approximately 1.15 times faster by median. The paths selected the same move on 11 of 12 positions. This small comparison measures latency and move agreement.

## Search quality floor

The project's locked two-second PUCT baseline uses a separate frozen checkpoint and search configuration. The optimization target is at least 10 times lower per-move latency on the same hardware while retaining playing quality, measured through position tests and paired direct games. Experiments have yet to establish both conditions. The final joint model awaits that promotion test. The [baseline record](SEARCH_BASELINE.md) specifies the comparison.
