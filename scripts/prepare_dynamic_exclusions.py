"""Build a small immutable mmap key index; never load millions into SQLite."""
import argparse,hashlib,json,sqlite3,sys,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import chess
from neurochess.encoding import position_key


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);arrays=[];sources=[]
    original=None
    if args.original:
        p=Path(args.original);a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.dtype!=np.uint8 or a.shape!=(7590978,16):raise ValueError('Original index shape/count differs from the prior novelty certificate')
        raw=a.copy().view('V16').reshape(-1)
        if len(np.unique(raw))!=7590978:raise ValueError('Original index contains duplicate keys')
        original=dict(path=str(p.resolve()),rows=len(raw),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),verification='User-downloaded original; exact count, uniqueness and local hash readback. Historical remote hash was not provided.')
        arrays.append(raw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for root in [ROOT/'data',ROOT/'output']:
            for p in root.rglob('*.npy'):
                if 'dynamic-curriculum-' in str(p) or 'report-app' in str(p):continue
                try:
                    a=np.load(p,mmap_mode='r',allow_pickle=False)
                    if not a.dtype.names or 'key' not in a.dtype.names:continue
                    arrays.append(a['key'].copy().view('V16').reshape(-1));sources.append(dict(path=str(p.relative_to(ROOT)),rows=len(a)))
                except (ValueError,OSError):continue
    for p in [ROOT/'output/curriculum-20261031/training-exclusions.fen']:
        if p.exists():arrays.append(np.array([position_key(chess.Board(l.strip())) for l in p.read_text().splitlines() if l.strip()],dtype='V16'))
    p=ROOT/'output/targeted-distill-20261102/frozen.json'
    if p.exists():arrays.append(np.array([bytes.fromhex(k) for k in json.loads(p.read_text())['keys']],dtype='V16'))
    keys=np.unique(np.concatenate(arrays));np.save(out/'known-prior-keys.npy',keys)
    source=ROOT/'output/targeted-scale-local-20261102/annotation.sqlite3';db=sqlite3.connect(f'file:{source}?mode=ro',uri=True)
    groups=[r[0] for r in db.execute("SELECT family FROM groups WHERE split!='train'")];db.close()
    (out/'frozen-groups.json').write_text(json.dumps(groups))
    proof=dict(unique_known_prior_keys=len(keys),frozen_source_groups=len(groups),files=sources,
        index_sha256=hashlib.file_digest((out/'known-prior-keys.npy').open('rb'),'sha256').hexdigest(),
        complete_original_7590978_key_index=original is not None,original_index=original,
        scope='Original historical core/human/fresh/repair/frozen key index when supplied, union all locally available newer labelled shards and frozen key/FEN lists. Exact-key exclusion does not prove isolation from historical games whose provenance is unavailable. Descendants are derived states, never counted as novel roots.')
    (out/'exclusion-audit.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='output/dynamic-curriculum-20261103');p.add_argument('--original');main(p.parse_args())
