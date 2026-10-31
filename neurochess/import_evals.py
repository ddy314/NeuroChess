"""Import Lichess eval-cache JSONL(.zst): white scores -> side-to-move targets."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
from urllib.request import urlopen
import chess
import chess.engine
import numpy as np
import zstandard
from .data import ShardWriter, record
from .encoding import position_key, split_for
from .telemetry import write_status
from .targets import score_targets, assign_targets, TARGET_SPEC

DEFAULT_URL = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'


def convert(entry, min_depth=20, min_knodes=50, min_pvs=2, temperature_cp=100):
    # Eval cache contains only four FEN fields; no invented repetition/clock supervision.
    fen = entry['fen'].split()
    if len(fen) != 4:
        raise ValueError('expected four-field eval-cache FEN')
    board = chess.Board(' '.join(fen) + ' 0 1')
    if not board.is_valid() or board.is_game_over() or board.legal_moves.count() < 2:
        return None, 'invalid_or_terminal'
    eligible = [e for e in entry.get('evals', []) if e.get('depth',0) >= min_depth
                and e.get('knodes',0) >= min_knodes and len(e.get('pvs',[])) >= min_pvs]
    if not eligible:
        return None, 'below_quality_threshold'
    # Never merge PVs of different searches/depths; choose one coherent MultiPV search.
    evaluation = max(eligible, key=lambda e: (e['depth'], e['knodes'], len(e['pvs'])))
    row, legal = record(board)
    count = assign_targets(row,board,legal,evaluation['pvs'],temperature_cp)
    return (board,row,evaluation['depth'],count),None



def exclusion_keys(files):
    keys = set()
    for filename in files:
        for line in Path(filename).read_text().splitlines():
            if line.strip():
                keys.add(position_key(chess.Board(line.strip())))
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=DEFAULT_URL)
    p.add_argument('--out', required=True)
    p.add_argument('--positions', type=int, default=1000000)
    p.add_argument('--max-lines', type=int, default=20000000)
    p.add_argument('--min-depth', type=int, default=20)
    p.add_argument('--min-knodes', type=int, default=50)
    p.add_argument('--min-pvs', type=int, default=2)
    p.add_argument('--temperature-cp', type=float, default=100)
    p.add_argument('--exclude-fens', nargs='*', default=[])
    p.add_argument('--status-dir')
    a = p.parse_args()
    if min(a.positions,a.max_lines,a.min_depth,a.min_knodes,a.min_pvs,a.temperature_cp) <= 0:
        p.error('positive limits required')
    writer = ShardWriter(a.out)
    reasons, depths, pv_counts, outcomes = Counter(), Counter(), Counter(), Counter()
    excluded = exclusion_keys(a.exclude_fens)
    digest = hashlib.sha256(); lines = accepted = 0; started = time.time(); last_log = 0
    source = urlopen(a.source, timeout=90) if a.source.startswith('https://') else open(a.source,'rb')
    headers = {k:source.headers.get(k) for k in ['ETag','Last-Modified','Content-Length']} if hasattr(source,'headers') else {}
    db = sqlite3.connect(Path(a.out)/'keys.sqlite')
    db.execute('CREATE TABLE positions (key BLOB PRIMARY KEY) WITHOUT ROWID')
    def progress(phase):
        info = dict(source=a.source, positions=accepted, target_positions=a.positions, scanned=lines,
                    rejected=dict(reasons), counts=writer.counts, seconds=time.time()-started)
        print(json.dumps(info),flush=True)
        if a.status_dir:
            Path(a.status_dir).mkdir(parents=True,exist_ok=True)
            write_status(a.status_dir,phase,note=f'引擎数据：{accepted}/{a.positions} 局面；扫描 {lines} 行', import_progress=info)
    try:
        reader = zstandard.ZstdDecompressor().stream_reader(source) if a.source.split('?')[0].endswith('.zst') else source
        with io.TextIOWrapper(reader, encoding='utf-8') as stream:
            for line in stream:
                lines += 1; digest.update(line.encode())
                try:
                    converted, reason = convert(json.loads(line),a.min_depth,a.min_knodes,a.min_pvs,a.temperature_cp)
                    if converted is None:
                        reasons[reason] += 1
                    else:
                        board,row,depth,pvs = converted; key=position_key(board)
                        if key in excluded:
                            reasons['evaluation_opening'] += 1
                        elif db.execute('INSERT OR IGNORE INTO positions VALUES (?)',(key,)).rowcount == 0:
                            reasons['duplicate_position'] += 1
                        else:
                            writer.add(split_for(board),row); accepted += 1
                            depths[depth] += 1; pv_counts[pvs] += 1
                            outcomes['white_to_move' if board.turn else 'black_to_move'] += 1
                            outcomes['win' if row['value'][0] > .75 else 'loss' if row['value'][2] > .75 else 'balanced'] += 1
                            if accepted % 4096 == 0: db.commit()
                except (ValueError,KeyError,TypeError,IndexError):
                    reasons['malformed_or_illegal'] += 1
                if time.time()-last_log >= 15:
                    progress('preparing');last_log=time.time()
                if accepted >= a.positions or lines >= a.max_lines:break
        if not all(writer.counts.values()):
            raise ValueError('Need nonempty train and validation splits; increase source sample')
        writer.close(dict(target_spec=TARGET_SPEC, kind='lichess-evals', args=vars(a), source=a.source, http=headers,
            target_reached=accepted>=a.positions, termination='target' if accepted>=a.positions else 'max_lines' if lines>=a.max_lines else 'source_eof',
            prefix_lines=lines, parsed_prefix_sha256=digest.hexdigest(), accepted=accepted,
            rejection_counts=dict(reasons), depth_histogram=dict(depths), pv_histogram=dict(pv_counts),
            distribution=dict(outcomes), score_perspective='white -> side-to-move',
            value_target='sf16 WDL at fixed ply=32, a score proxy NOT observed or calibrated game outcome',
            q_target='W-L of same proxy; only annotated legal first moves',
            policy_target='softmax of side-to-move centipawns at temperature_cp; mate outranks cp',
            history='unknown clocks/repetition: train with --history-free',
            split='position-hash-1-percent; compatible with phase1',
            sampling='filtered stream prefix, NOT uniform sample of complete database',
            excluded_openings=len(excluded), license='CC0-1.0'))
        progress('prepared')
    except BaseException:
        progress('prepare_failed');raise
    finally:
        db.commit();db.close();source.close()


if __name__ == '__main__': main()
