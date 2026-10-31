"""Build the upload bundle from already prepared data and phase-1 weights."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent.parent
out=root/'output/phase2';out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((root/'data/evals-v1/manifest.json').read_text())
if sum(manifest['counts'].values()) < 800_000:
    raise RuntimeError('Expected at least 800000 verified positions')
files={}
for folder in ['neurochess','configs','scripts']:
    for path in (root/folder).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc','.so'}:
            files[str(path.relative_to(root))]=path
files['pyproject.toml']=root/'pyproject.toml'
for path in (root/'data/evals-v1').glob('*'):
    if path.suffix in {'.npy','.json'}:files[str(path.relative_to(root))]=path
files['phase2-init.pt']=out/'phase2-init.pt'
files['phase2-exclude.fen']=out/'exclude.fen'
checksums={}
for name,path in files.items():
    with path.open('rb') as stream:checksums[name]=hashlib.file_digest(stream,'sha256').hexdigest()
check=out/'phase2-checksums.json';check.write_text(json.dumps(checksums,indent=2))
files[check.name]=check
archive=out/'neurochess-phase2.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for name,path in sorted(files.items()):tar.add(path,arcname=name)
print(json.dumps({'bundle':str(archive),'bytes':archive.stat().st_size,'files':len(files)},indent=2))
