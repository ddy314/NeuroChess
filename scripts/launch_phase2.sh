#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
session="${NC_SESSION:-neurochess-engine-v1}"
bundle=output/phase2/neurochess-phase2.tar.gz
[[ -s "$bundle" ]] || { echo 'Missing prepared bundle; run scripts/package_phase2.py first.' >&2; exit 1; }
if [[ "${NC_REUSE_SESSION:-0}" != 1 ]]; then
  colab new -s "$session" --gpu "${NC_GPU:-L4}"
fi
parts=$(mktemp -d)
trap 'rm -rf "$parts"' EXIT
split -b 32M -d -a 3 "$bundle" "$parts/phase2-part-"
for chunk in "$parts"/phase2-part-*; do
  colab upload -s "$session" "$chunk" "/content/$(basename "$chunk")"
done
colab exec -s "$session" --file scripts/phase2_assemble.py --timeout 180
# Drive consent must be completed by the account owner.
colab drivemount -s "$session" /content/drive
# The Python launcher independently checks os.path.ismount before starting work.
colab exec -s "$session" --file scripts/phase2_colab.py --timeout 86400
