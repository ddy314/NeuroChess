# Lichess three-head guarded single epoch (2026-11-04)

## Verified outcome as of 2026-11-06

The main run was explicitly paused at update4000 after3,072,000 new rows and
1,024,000 replay presentations; it did **not** complete the9.8M-row epoch.
A/B/C each completed500 controlled updates from that same checkpoint and
shared the same next384,000 new rows; none was automatically selected.
The local D diagnostic did not trigger training. Subsequent512-position
search auditing failed the original search screen. Independent repair from
the frozen parent stopped at50/100 updates on its search gate. Original test
labels remained unused and the baseline was not promoted away. See
[A/B/C](LICHESS_4000_ABLATION.md), [search/repair](LICHESS_SEARCH_REPAIR.md)
and [cumulative verified progress](../reports/TRAINING_PROGRESS.md).

The remainder records the original epoch protocol; its final full-validation/
test procedure was not executed at the authorized mid-run pause. Historical
runtime directories/dashboard URLs do not establish a currently active VM.

## Original training protocol

Initialize from the exact immutable `puct-2s-v1` checkpoint SHA in
`configs/search-baseline-lock.json` (155,000,581 parameters, one shared encoder).
The PAWN candidate and its double-encoder parent are not initializers for this run.
Policy, annotated-action Q and WDL value all train jointly with the shared trunk.

One epoch means exactly 9,800,000 new training rows, each once, including the final
partial batch. The 100k validation and 100k test rows do not train. Each update
has up to 768 new rows plus 256 old-core replay rows (25% of presentations).
Replay is additional, separately counted, never described as extra new epochs.
The pilot's rows count toward the same traversal. No unconditional second epoch
or automatic retry after a guard stop is authorized by this plan.

Trial 2 peak LR 3e-7, 100-update warmup, cosine decay to 3e-8; AdamW, BF16, gradient
clipping at 1.0. Loss is policy CE + 0.5 masked Q MSE + value CE, normalized by
available labels over the entire effective batch, independent of microbatch/tail
size. Value supervision assigns equal total mass to winning/neutral/losing
strata within the effective batch; evaluation remains at the original distribution.
On replay inputs, an immutable parent anchors policy KL, legal-action Q MSE
and WDL KL (weights 0.5/0.25/4). These preservation terms are not new data labels.
Replay excludes all new validation/test keys, old reference validation keys and
all frozen regression roots. Unknown history remains masked.

Validate at initialization, pilot update 100, every 250 updates and the exact
end. Use seeded all-shard samples of 8192 new validation rows and 8192 old-core
validation rows plus the complete fixed historical general-validation array.
Monitor all three heads individually, and winning/neutral/losing value strata.
For each old monitor: loss/policy CE <= initial +1%; value CE <= +2%; top1 drop
<=0.5 percentage points; Q/value MAE increases <=0.005; stratum MAE increase
<=0.01. New loss <= +1%, top1 drop <=0.5 points, Q/value MAE increases <=0.01,
stratum MAE increase <=0.015. Any violation stops rather than merely logging.

At initialization, pilot, every 2000 updates and the final update, also compare
48 frozen development roots using the locked two-second PUCT on the same GPU,
fresh Stockfish 1M-node candidate-union labels, and the same-GPU parent choices.
Reject new >=100cp mistakes worsening parent by >=75cp, or mean regret delta
>10cp. These finite teachers and development roots are fallible monitoring
screens; they are not independent Elo tests or universal non-regression proofs.

`last.pt` is the latest metric-passing resumable model/optimizer/RNG/cursor.
`best.pt` is the lowest new-validation loss among checkpoints also passing the
search screen; initialization is the fallback. A failed candidate is saved as
`rejected.pt`, never silently replacing best. SHA readback follows committed
checkpoints. Resume checks the unchanged plan, restored optimizer/RNG, and exact
new-data cursor; it does not restart the epoch or reset Adam.

After training or a guard stop, select best, recheck validation, then evaluate
full 100k validation and 100k test once. Persist completion, exact presented-row
counts, gate reports, best/last/selected weights, source, and readback hashes;
release the one L4 instance. Failures persist diagnostics before release.
A guard stop is explicitly an incomplete epoch, never labelled successful
one-epoch completion. No automatic desktop default/baseline promotion occurs.

The first 100-update pilot was discarded: aggregate metrics hid worsening
winning-value errors (old general 0.22791 -> 0.24284, old core 0.29599 -> 0.31371,
new 0.27630 -> 0.29711). A NumPy boolean serialization bug interrupted guard
reporting; it was fixed and tested. Trial 2 restarts from the same frozen parent
with a lower LR, balanced value supervision and stronger frozen-value anchoring.
No guard threshold was relaxed. The failed pilot is a separate discarded
lineage, never counted as one epoch in the accepted candidate.
