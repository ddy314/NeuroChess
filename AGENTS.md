# Search optimization constraints

For chess search changes, follow `docs/SEARCH_BASELINE.md`. The user-fixed floor
is the final optimized two-second PUCT in `configs/search-baseline-lock.json`.
Do not silently replace its weights/configuration or promote a weaker variant.
The active optimization target is at least10x less per-move latency on the same
hardware while retaining baseline playing quality. Report achieved speed and
quality separately; never claim success from fewer simulations or NPS alone.
