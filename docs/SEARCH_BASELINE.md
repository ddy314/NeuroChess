# Search performance floor

As of2026-11-06 the floor remains unchanged. Native/quantized fast candidates,
PAWN, Lichess4000, A/B/C, manual weight interpolations and the50-step repair
have not replaced it. The joint10x-latency-and-preserved-quality objective
remains unmet; see `reports/TRAINING_PROGRESS.md` for the separate outcomes.

The user designated the final optimized two-second PUCT as the minimum quality
baseline on2026-10-31. Its locked configuration, checkpoint SHA256 and source
fingerprints are in `configs/search-baseline-lock.json`; the independent frozen
checkpoint/source copy is under `output/puct-10x-20261031/frozen-baseline/`.
The reference implementation includes the pending-loss batching fix, neural
input cache, exact-history subtree reuse, and elapsed-time root pruning.

The current target is at least10x lower single-game latency: approximately0.2s
per move rather than2s on the same local RTX4060Laptop. Keep model initialization,
compilation and warmup separate, but report them. P95 is the primary latency
criterion; also report maximum latency and actual neural inputs. Cross-game
throughput and repeated terminal backups cannot substitute for move latency.

Do not overwrite or weaken the baseline to make an experiment pass. Research
variants may fail; do not promote them when quality degrades. Compare against
the frozen final PUCT, not the older512-simulation player. Use frozen tactical
and position regressions, fresh game-grouped heldouts and color-paired direct
play, with uncertainty. Finite tests do not prove universal non-regression.
Reducing search alone does not establish the requested quality-preserving gain.
Record explicitly if the10x target or the quality requirement remains unmet.
