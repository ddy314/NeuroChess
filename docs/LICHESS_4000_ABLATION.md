# Lichess step-4000 controlled continuation

## Completed outcome (2026-11-06 documentation snapshot)

All A/B/C branches completed500 updates and passed their unchanged supervised
and original48-root screens. Each used the same384,000 new rows and128,000
replay presentations, not three disjoint new datasets. B's1.5x LR did not
materially improve final new loss over A; C improved the new Value metric
modestly without establishing playing-strength improvement. Main remains
logically paused at4000; no winner or default model was selected. Later local
D16-batch diagnosis did not meet its predeclared trigger and trained0 updates.
Later512-position auditing and independent50-step repair are separate trials.

Evidence: [branches](../reports/evidence/research-progress-20261106/lichess/ablation-results.json),
[gradients](../reports/evidence/research-progress-20261106/lichess/gradient-diagnosis.json),
[D decision](../reports/evidence/research-progress-20261106/lichess/local-d-results.json).
The execution protocol below describes the then-existing L4; its notebook,
kernel and process references are not a current service-status claim.

## Execution protocol

The user authorized pausing the main single-epoch run after its step-4000
metric/search gates and checkpoint readback, then diagnosing gradients and
running A/B/C for at most 500 updates each. Original thresholds and the frozen
search floor remain authoritative. A guard failure stops that branch early.
There is no automatic winner selection, model promotion, or main continuation.

`scripts/watch_lichess_epoch_ablation.py` observes the original checkpoint
certificate, checks the PID identity, sends SIGSTOP only after the step-4000
certificate exists, and preserves immutable checkpoint inodes. The current
main process must remain stopped; do not send SIGCONT after branch experiments.
Its notebook kernel remains busy waiting on the stopped child while the
experiment runs independently in the same mounted L4 instance.

`scripts/lichess_epoch_ablation.py` and `neurochess/epoch_ablation.py` restore
the exact model, copied Adam state, RNG, and new/replay data cursor for each
branch. The deterministic data sequence is hashed and checked at shared
100/250/500-update boundaries. Every branch trains all three heads.

| Branch | Learning-rate multiplier | Value balancing strength |
| --- | --- | --- |
| A | 1.0 | 1.0 |
| B | Ramp from 1.0 to 1.5 in 50 updates | 1.0 |
| C | 1.0 | 0.5 |

The absolute cosine schedule continues from update 4000; Adam and warmup are
not reset. C blends the original confidence weights with fully balanced
weights; it does not change labels. Replay fraction and all retention
coefficients remain unchanged. Eight effective batches diagnose the exact
weighted shared-trunk gradients without optimizer steps. Gradient cosines do
not establish a causal mechanism or reproduce Adam's preconditioned direction.

Each branch is checked at offsets 100, 250, and 500 against the original
frozen-parent validation metrics, with the original locked 2-second search
screen at offsets 100 and 500. The initial comparison populations are the
same new 8192, old-core 8192, and old-general 4096 positions. Full test labels
remain sealed. Complete branches retain aligned per-position development
metrics and position-paired bootstrap comparisons against A. These intervals
are neither game-clustered nor independent confirmation after tuning.

Checkpoints and evidence are copied to the existing Drive mount and SHA
read back. The original main checkpoint is verified unchanged. Candidate
checkpoints include a distinct experimental plan hash so the original trainer
cannot silently resume them with the wrong schedule or balancing settings.

`scripts/collect_lichess_ablation.py` uses existing Colab CLI credentials to
copy small result/evidence files to the local workspace, verifies their
manifest, and acknowledges the result digest. The coordinator releases the
L4 only after persistent result verification, allowing a bounded report-copy
window. It leaves the main logical training run paused at 4000 for the user's
next decision. A Codex heartbeat reports completion or actionable failure;
unchanged progress stays quiet. No additional VM or Drive login is required
between the diagnostic and branches.

The historical deployment and status routes are recorded in
`output/lichess-joint-epoch1-20261104/auto4000/HANDOFF.json`.
