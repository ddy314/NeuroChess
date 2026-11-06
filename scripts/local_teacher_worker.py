"""CPU-only resident Stockfish workers; deliberately never imports torch."""
import threading
import chess
import chess.engine
from neurochess.encoding import position_key
from scripts.targeted_mistakes import classify, stable_labels, needs_repair

ENGINE = None


class CandidateError(ValueError):
    """Invalid source position, quarantined rather than retried as teacher work."""


def candidate_issue(job):
    headers = job.get('source_game_headers', {})
    if headers.get('Variant', 'Standard') != 'Standard':
        return 'nonstandard_variant'
    # This corpus promises complete elite standard games from the normal start.
    # Broadcasts can incorrectly call Chess960 "Standard" in their PGN headers.
    if job['initial_fen'] != chess.STARTING_FEN:
        return 'nonstandard_initial_position'
    return None


def initialize(binary):
    global ENGINE
    ENGINE = chess.engine.SimpleEngine.popen_uci(binary, timeout=120)
    ENGINE.configure({'Threads': 1, 'Hash': 64, 'Skill Level': 20,
                      'UCI_LimitStrength': False})
    threading._register_atexit(ENGINE.quit)


def board_for(job):
    if issue := candidate_issue(job):
        raise CandidateError(issue)
    board = chess.Board(job['initial_fen'])
    try:
        for move in job['history']:
            board.push_uci(move)
    except ValueError as error:
        raise CandidateError(f'Illegal source history: {error}') from error
    if position_key(board).hex() != job['key']:
        raise CandidateError('Node key/history mismatch')
    if not board.is_valid() or board.is_game_over(claim_draw=True):
        raise CandidateError('Invalid or terminal source position')
    return board


def label(task):
    job, pred = task
    board = board_for(job)

    def analyze(nodes, moves=None):
        options = {} if moves is None else dict(root_moves=[board.parse_uci(m) for m in moves])
        infos = ENGINE.analyse(board, chess.engine.Limit(nodes=nodes),
                               multipv=min(4, board.legal_moves.count()) if moves is None else len(moves),
                               game=object(), **options)
        result = []
        for info in infos:
            score = info['score'].white()
            pv = dict(line=' '.join(m.uci() for m in info['pv']),
                      depth=info.get('depth'), nodes=info.get('nodes'))
            pv['mate' if score.is_mate() else 'cp'] = score.mate() if score.is_mate() else score.score()
            result.append(pv)
        return result

    if pred is None:
        return dict(job=job, prediction=None, teacher=analyze(100000), stable=None, repair=False)
    coarse = analyze(10000)
    moves = list(dict.fromkeys([p['line'].split()[0] for p in coarse] + pred['top_moves']))
    first = analyze(30000, moves)
    decision = classify(first, pred['move'], pred['value'], board.turn)
    deep, stable = None, None
    if needs_repair(decision):
        first = analyze(100000, moves)
        deep = analyze(300000, moves)
        stable = stable_labels(first, deep, board.turn)
        decision = classify(deep, pred['move'], pred['value'], board.turn)
    return dict(job=job, prediction=pred, teacher=deep or first, teacher_previous=first,
                stable=stable, decision=decision,
                repair=bool(needs_repair(decision) and stable), teacher_move=decision['best'])
