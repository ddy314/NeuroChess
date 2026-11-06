"""Strict, history-free Lichess evaluation conversion and auditable labels."""
from collections import Counter
import hashlib
import json

import chess
import numpy as np

from .data import DTYPE, record
from .encoding import position_key
from .targets import TARGET_SPEC, assign_targets, score_targets

CLEANING_SPEC = dict(
    schema=1, min_depth=24, min_knodes=1000, min_pvs=2,
    full_selected_pvs_legal=True, distinct_legal_root_moves=True,
    unknown_history='mask meta indices 5,6,7,10; never infer repetitions/clocks',
    global_value_max_wdl_difference=0.10,
    global_value_max_cp_difference=100,
    different_best_move_max_cp_difference=25,
    different_best_move_max_wdl_difference=0.05,
    target_spec=TARGET_SPEC,
    reference='highest depth then knodes among all threshold-qualified evaluations',
    selected='highest depth then knodes then PV count among qualified MultiPV searches',
    split='blake2b canonical position key; little-endian first 8 bytes mod 100: 0 val, 1 test, else train',
    independence='exact canonical position only; source game IDs/history unavailable',
)


WORKER_EXCLUSIONS = None


def init_clean_worker(path):
    global WORKER_EXCLUSIONS
    WORKER_EXCLUSIONS = np.load(path, mmap_mode='r', allow_pickle=False)


class Rejected(ValueError):
    pass


def reject(reason):
    raise Rejected(reason)


def split_key(key):
    bucket = int.from_bytes(key[:8], 'little') % 100
    return 'val' if bucket == 0 else 'test' if bucket == 1 else 'train'


def qualified(e):
    if not isinstance(e, dict):
        return False
    d, n = e.get('depth'), e.get('knodes')
    return (type(d) is int and type(n) is int and d >= 24 and n >= 1000
            and isinstance(e.get('pvs'), list) and len(e['pvs']) >= 1)


def checked_pvs(board, evaluation):
    first = []; ranks = []; values = []; cps = []
    for pv in evaluation['pvs']:
        if not isinstance(pv, dict):
            reject('malformed_score_or_pv')
        try:
            rank, value, _ = score_targets(pv, board.turn)
        except (ValueError, TypeError):
            reject('malformed_score_or_pv')
        if 'cp' in pv and abs(pv['cp']) > 50000:
            reject('score_out_of_range')
        if 'mate' in pv and abs(pv['mate']) > 10000:
            reject('score_out_of_range')
        line = pv.get('line')
        if not isinstance(line, str) or not line.split():
            reject('empty_pv')
        variation = board.copy(stack=False)
        initial = None
        for i, uci in enumerate(line.split()):
            if uci == '0000':
                reject('null_move_in_pv')
            try:
                move = variation.parse_uci(uci)
            except ValueError:
                reject('illegal_pv_continuation')
            if not i:
                initial = move
            variation.push(move)
        if initial in first:
            reject('duplicate_pv_root_move')
        first.append(initial); ranks.append(rank); values.append(value)
        cps.append(pv.get('cp'))
    if ranks[0] != max(ranks):
        reject('pv1_not_best_score')
    return first, ranks, values, cps


def strict_convert(entry, excluded=None):
    """Return one coherent MultiPV target; reject global/value contradictions."""
    if not isinstance(entry, dict) or not isinstance(entry.get('evals'), list):
        reject('malformed_record')
    candidates = [e for e in entry['evals'] if qualified(e)]
    multipv = [e for e in candidates if len(e['pvs']) >= 2]
    if not multipv:
        reject('below_depth_knodes_or_multipv')
    fen = entry.get('fen')
    if not isinstance(fen, str) or len(fen.split()) != 4:
        reject('malformed_fen')
    try:
        board = chess.Board(fen + ' 0 1')
    except ValueError:
        reject('malformed_fen')
    if not board.is_valid() or board.is_game_over() or board.legal_moves.count() < 2:
        reject('invalid_terminal_or_forced_root')
    if excluded is not None and len(excluded):
        key = np.void(position_key(board))
        where = int(np.searchsorted(excluded, key))
        if where < len(excluded) and excluded[where] == key:
            reject('prior_or_heldout_overlap')
    chosen = max(multipv, key=lambda e: (e['depth'], e['knodes'], len(e['pvs'])))
    reference = max(candidates, key=lambda e: (e['depth'], e['knodes']))
    moves, ranks, values, cps = checked_pvs(board, chosen)
    if reference is chosen:
        ref_moves, ref_values, ref_cps = moves, values, cps
    else:
        # Only reference PV1 is needed; each selected MultiPV is checked in full.
        ref_moves, _, ref_values, ref_cps = checked_pvs(board, dict(reference, pvs=reference['pvs'][:1]))
    gap = abs(values[0] - ref_values[0])
    if gap > CLEANING_SPEC['global_value_max_wdl_difference'] + 1e-12:
        reject('global_value_disagreement')
    same_kind = ('mate' in chosen['pvs'][0]) == ('mate' in reference['pvs'][0])
    if not same_kind:
        reject('mate_cp_reference_disagreement')
    cp_gap = abs(cps[0] - ref_cps[0]) if cps[0] is not None and ref_cps[0] is not None else None
    if cp_gap is not None and cp_gap > CLEANING_SPEC['global_value_max_cp_difference']:
        reject('global_cp_disagreement')
    if moves[0] != ref_moves[0]:
        if cp_gap is None or cp_gap > 25 or gap > 0.05 + 1e-12:
            reject('global_best_move_disagreement')
    row, legal = record(board)
    assign_targets(row, board, legal, chosen['pvs'])
    row['meta'][[5, 6, 7, 10]] = 0
    evidence = dict(fen=fen, canonical_fen=' '.join(board.fen().split()[:4]),
                    key=position_key(board).hex(), evaluation=chosen,
                    reference_pv1=dict(reference, pvs=reference['pvs'][:1]),
                    global_value_gap=gap, global_cp_gap=cp_gap)
    return row, evidence


def convert_batch(payload):
    frame_offset, lines = payload
    rows = []; evidence = []; counts = Counter()
    for line in lines:
        counts['scanned'] += 1
        try:
            entry = json.loads(line)
            row, item = strict_convert(entry, WORKER_EXCLUSIONS)
        except Rejected as error:
            counts[str(error)] += 1
            continue
        except (ValueError, KeyError, TypeError, IndexError):
            counts['malformed_record'] += 1
            continue
        item['source_frame_offset'] = frame_offset
        item['source_json_sha256'] = hashlib.sha256(line).hexdigest()
        rows.append(row); evidence.append(item)
    return np.array(rows, dtype=DTYPE), evidence, dict(counts)


def audit_rows(rows):
    if rows.dtype != DTYPE:
        raise ValueError('Wrong training row schema')
    legal = rows['moves'] >= 0
    if (not np.all(legal.sum(-1) >= 2) or not np.isfinite(rows['meta']).all()
            or np.any(rows['meta'][:, [5, 6, 7, 10]])
            or not np.isfinite(rows['policy']).all() or np.any(rows['policy'] < 0)
            or not np.allclose(rows['policy'].sum(-1), 1, atol=.002)
            or np.any(rows['policy'][~legal]) or np.any(rows['qmask'][~legal])
            or not np.all(rows['qmask'].sum(-1) >= 2)
            or not np.isfinite(rows['q']).all() or np.any(abs(rows['q']) > 1)
            or not np.isfinite(rows['value']).all() or np.any(rows['value'] < 0)
            or not np.allclose(rows['value'].sum(-1), 1, atol=.002)
            or not np.all(rows['value_weight'] == 1)):
        raise ValueError('Invalid masks, normalized targets, values or history')
    for row in rows:
        ids = row['moves'][row['moves'] >= 0]
        if len(ids) != len(np.unique(ids)):
            raise ValueError('Duplicate legal action IDs')
