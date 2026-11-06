"""Export small, path-normalized research records, never weights or datasets.

The source files remain in ignored local experiment directories. Provenance
records both the original and exported SHA256; numbers are not recomputed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


RECORDS = {
    'joint/exposure.json': 'output/joint-deep-analysis-20261103/checkpoint-and-exposure.json',
    'joint/data-audit.json': 'output/joint-deep-analysis-20261103/data-audit.json',
    'joint/search-assessment.json': 'output/joint-deep-analysis-20261103/search-assessment.json',
    'joint/weight-movement.json': 'output/joint-deep-analysis-20261103/weight-movement-summary.json',
    'dynamic/candidate-audit.json': 'output/dynamic-curriculum-20261103/new-elite/full-audit.json',
    'dynamic/candidate-manifest.json': 'output/dynamic-curriculum-20261103/new-elite/candidate-manifest.json',
    'pawn/run-summary.json': 'output/pawn-20m-20261103/run-summary.json',
    'pawn/arena-analysis.json': 'output/pawn-arena-20261104/analysis.json',
    'pawn/critical-confirmation.json': 'output/pawn-arena-20261104/critical-confirmation.json',
    'pawn/search-work-analysis.json': 'output/pawn-arena-20261104/search-work-analysis.json',
    'lichess/cleaning-completion.json': 'data/lichess-depth24-knodes1000-clean-10m-20261104/completion.json',
    'lichess/ablation-results.json': 'output/lichess-joint-epoch1-20261104/auto4000/results/results.json',
    'lichess/gradient-diagnosis.json': 'output/lichess-joint-epoch1-20261104/auto4000/results/gradient-diagnosis.json',
    'lichess/collector-complete.json': 'output/lichess-joint-epoch1-20261104/auto4000/results/collector-complete.json',
    'lichess/local-d-results.json': 'output/lichess-local-d-20261105/results.json',
    'lichess/search-audit-512.json': 'output/lichess-search-audit-512-20261105/results.json',
    'lichess/failure-mechanisms.json': 'output/lichess-five-failure-mechanisms-20261105/final-analysis.json',
    'lichess/repair-plan.json': 'output/lichess-five-failure-mechanisms-20261105/repair-plan.json',
    'lichess/manual-weight-probes.json': 'output/lichess-five-failure-mechanisms-20261105/manual-weight-probes/results.json',
    'repair100/results.json': 'output/lichess-local-repair-pilot-20261105/results.json',
    'repair100/data-ready.json': 'output/lichess-local-repair-pilot-20261105/data-ready.json',
    'repair100/initial-validation.json': 'output/lichess-local-repair-pilot-20261105/initial-validation.json',
    'repair100/parameter-movement-step5.json': 'output/lichess-local-repair-pilot-20261105/parameter-movement-step5.json',
    'repair100/verification.json': 'output/lichess-local-repair-pilot-20261105/delivery-verification.json',
    'repair100/data-checkpoint-verification.json': 'output/lichess-local-repair-pilot-20261105/delivery-data-checkpoint-verification.json',
    'repair100/interpretation.json': 'output/lichess-local-repair-pilot-20261105/delivery-summary.json',
}
for index in [147, 296, 302, 386, 420]:
    RECORDS[f'lichess/coherent-teachers/{index}.json'] = (
        f'output/lichess-five-failure-mechanisms-20261105/coherent-teachers/{index:03d}.json'
    )


def normalize(value, root):
    if isinstance(value, dict):
        return {normalize(k, root): normalize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, root) for v in value]
    if isinstance(value, str):
        value = value.replace(str(root), '{repo}')
        value = value.replace('/content/drive/MyDrive/', '{private-drive}/')
        return re.sub(r'/home/[^/\s]+', '{home}', value)
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(root, out):
    entries = []
    for name, source in RECORDS.items():
        original = (root / source).read_bytes()
        normalized = normalize(json.loads(original), root)
        blob = (json.dumps(normalized, ensure_ascii=False, indent=2) + '\n').encode()
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        entries.append(dict(file=name, source=source, source_sha256=digest(original),
                            exported_sha256=digest(blob)))
    manifest = dict(schema=1, as_of='2026-11-06', records=entries,
        transformations=['Normalize repository/home/private Drive path strings recursively; preserve numeric values and outcome fields.'],
        scope='Curated experiment summaries and evaluation evidence, not training data or an artifact bundle.',
        excluded=['weights', 'optimizer tensors', 'training datasets', 'runtime logs', 'credentials', 'handoffs'],
        lichess_original_test_labels_used=False, automatic_model_promotion=False)
    (out / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / 'reports/evidence/research-progress-20261106'
    result = export(root, out)
    print(json.dumps(dict(records=len(result['records']), out=str(out))))


if __name__ == '__main__':
    main()
