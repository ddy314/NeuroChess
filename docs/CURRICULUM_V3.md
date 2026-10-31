# Curriculum v3

Continue from engine-v2-expanded/last.pt (cumulative optimizer step 7500) with its optimizer state.

- Stage A: 4000 updates on 5M engine positions with policy-only human replay.
- Cloud CPU teacher: SF18, 16 workers across two 8-core CPU runtimes, 20k-node coarse MultiPV4 and one coherent 200k-node search over teacher/student candidate union. 200k training and 2k validation positions selected from the existing corpus; these are reannotations, not novel positions relative to core.
- Publish four immutable cumulative waves, adding 50k training and 500 validation rows each. Audit each wave and verify the persistent archive SHA before publishing its ready marker.
- Four subsequent stages, 1000 updates each, mix core/hard/human policy batches 70/20/10; exclude each wave's training keys from old-label core. Learning rates 8e-6, 6e-6, 4e-6, 3e-6.
- Total: 8000 additional updates, reaching cumulative step 15500.
- Unified labels: side-to-move centipawn policy softmax with temperature 100 cp; sf16 WDL conversion at fixed ply 32; Q=W-L only on annotated actions.
- Fixed original 8457-position reference validation selects best stage checkpoint. Final evaluation uses 32 frozen opening pairs (absent from core), SF18 at 1000 nodes/Threads1/Skill20/no strength limit, direct 64 games and 512-simulation multihead PUCT 64 games.
- The two teacher partitions and GPU execute on separate cloud runtimes, using VM-local files and atomic persistent Drive delivery. The second teacher handles ordered jobs [101000,202000), verifies and persists its SQLite snapshot, then releases itself. The primary preserves its existing labels, handles [0,101000), merges only an exact disjoint helper scope with the identical label specification, publishes all four audited waves, then releases itself. GPU releases only after final training, legal terminal PGNs and aggregate scores for all 128 games, full artifact readback, and checkpoint/optimizer/data hash verification.

Entrypoints: `scripts/teacher_colab.py`, `scripts/curriculum_colab.py`.
