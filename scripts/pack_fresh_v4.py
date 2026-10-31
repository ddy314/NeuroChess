"""Freeze small code/assets bundle; never include weights, credentials or datasets."""
from pathlib import Path
import argparse,hashlib,json,tarfile
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();archive=Path(a.out)
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for folder in ['neurochess','scripts','configs']:
  for f in sorted(Path(folder).rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.so']:tar.add(f,arcname=str(f))
 for name in ['eval-openings.fen','training-exclusions.fen']:
  tar.add(Path('output/curriculum-20261031')/name,arcname='fresh-v4-assets/'+name)
proof=dict(path=str(archive),sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),bytes=archive.stat().st_size)
archive.with_suffix('.sha256.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
