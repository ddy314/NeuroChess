"""Assemble a chunked upload without holding the archive in RAM."""
from pathlib import Path
import shutil
root=Path('/content')
parts=sorted(root.glob('phase2-part-[0-9][0-9][0-9]'))
if not parts:
    raise RuntimeError('No uploaded bundle parts')
expected=[f'phase2-part-{i:03d}' for i in range(len(parts))]
if [p.name for p in parts]!=expected:
    raise RuntimeError('Noncontiguous upload parts')
with (root/'neurochess-phase2.tar.gz.tmp').open('wb') as out:
    for part in parts:
        with part.open('rb') as source:shutil.copyfileobj(source,out,8*1024*1024)
(root/'neurochess-phase2.tar.gz.tmp').replace(root/'neurochess-phase2.tar.gz')
print('Assembled',len(parts),'parts;', (root/'neurochess-phase2.tar.gz').stat().st_size,'bytes')
