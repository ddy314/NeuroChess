# Lichess 10M preparation (2026-11-04)

Completed:10,000,000 unique positions (9.8M train/100k val/100k test), zero
reported prior/heldout, duplicate or cross-split position-key overlap. Every
training row passed numeric audit, selected PVs were replayed before publication,
and9,932 retained rows were reconstructed. See the [completion certificate](../reports/evidence/research-progress-20261106/lichess/cleaning-completion.json).
This certifies preparation, not a completed epoch or game independence.
The protocol below records the completed preparation.

The authorized job prepares exactly 10,000,000 new, unique positions: 9,800,000
train, 100,000 validation and 100,000 test. It runs on one high-memory CPU VM;
it does not start training or replace the locked playing baseline.

- A single search must meet depth >= 24 and knodes >= 1000 and contain at least
  two distinct legal root moves. Searches are never merged to manufacture MultiPV.
- Select the qualified MultiPV search with highest depth, then knodes, then PV
  count. Replay every move of every selected PV; reject null moves, invalid FENs,
  terminal/forced roots, duplicate first moves and malformed scores.
- Compare its best score to PV1 of the highest-depth qualified search, including
  single-PV searches. Reject W-L proxy difference > 0.10, centipawn difference
  > 100, or mate/centipawn disagreement. Different best moves require a near tie
  (<= 25 cp and <= 0.05 W-L difference). These are consistency filters, not a
  substitute for independent engine reanalysis or a proof of label accuracy.
- Keep white-to-mover conversion, annotated-action-only Q masks and the existing
  versioned score-proxy target definition. Explicitly zero unknown clock,
  repetition and move-number features. No fabricated game-result labels.
- Exact canonical 128-bit keys exclude certified historical keys, newer local
  structured shards/candidate states, retained FEN/PGN positions, all 20M prior
  PAWN training roots and every position of its official val/test games. An HTML
  error response misnamed `.pgn` is recorded as non-PGN rather than parsed.
- SQLite enforces global uniqueness. Position hashes assign disjoint splits.
  The source has no original game IDs or complete histories: whole-game
  independence and historical source-game independence cannot be established.
- Sample 128 compressed-byte strata with accepted split quotas and seeded cyclic starts. Redistribute
  any deficit from an exhausted stratum across the remaining strata. This gives broad source coverage, not a uniform sample of the
  complete position population. No artificial score/material balance is imposed;
  accepted depth, PV, material, side-to-move and value profiles are retained.
- Pin the source URL, ETag and byte size. Validate every HTTP range; carry JSON
  fragments across zstd frames, including pzstd skippable metadata.
- Publish paired compressed training and source-evidence shards using SHA256
  readback. Update `ready.json` only after both files are durable. Resume rebuilds
  SQLite from committed shards, ignoring orphaned partial publications.
- Final checks cover every target row, all persisted file hashes, exact counts,
  prior-key exclusion and global/cross-split uniqueness. Reconstruct a fixed
  subset from its retained evidence. Publish `complete=true` only after passing.

Implementation: `neurochess/lichess_clean.py`, `scripts/prepare_lichess_cloud.py`.
Local plan, launch record and evidence: `output/lichess-clean-10m-20261104/`.
Persistent destination:
`MyDrive/neurochess-data/lichess-depth24-knodes1000-clean-10m-20261104`.
The job releases its VM after verified completion, or after persisting a failure
record. The same saved plan and source version can resume committed work.
