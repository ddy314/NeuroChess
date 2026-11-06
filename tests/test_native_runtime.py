"""Optional real native executable checks; Python is only the test controller."""
import json
import os
import queue
import subprocess
import threading
import time

import chess
import chess.engine
import pytest


@pytest.fixture(scope='module')
def command():
    binary = os.environ.get('NEUROCHESS_NATIVE_BINARY')
    plan = os.environ.get('NEUROCHESS_NATIVE_PLAN')
    if not binary or not plan:
        pytest.skip('Set native executable and TensorRT plan to run GPU integration checks')
    return [binary, '--engine', plan, '--uci', '--history-free']


def telemetry(result):
    return json.loads(result.info['string'].split('telemetry ', 1)[1])


def test_fixed_budget_full_history_reuse_and_history_identity(command):
    with chess.engine.SimpleEngine.popen_uci(command, timeout=30) as engine:
        board, game = chess.Board(), object()
        for _ in range(4):
            result = engine.play(board, chess.engine.Limit(nodes=128), game=game, info=chess.engine.INFO_ALL)
            assert result.move in board.legal_moves
            data = telemetry(result)
            assert data['native']['roots'][0]['new_simulations'] == 128
            assert data['requested_positions'] == data['network_inputs'] + data['cache_hits'] + data['batch_duplicates']
            board.push(result.move)
        assert data['native']['stats']['tree_reuses'] == 3
        # Keep the game identifier, but remove the actual history: same FEN
        # cannot inherit history-dependent proofs from the previous tree.
        fresh = chess.Board(board.fen(en_passant='fen'))
        result = engine.play(fresh, chess.engine.Limit(nodes=64), game=game, info=chess.engine.INFO_ALL)
        root = telemetry(result)['native']['roots'][0]
        assert root['retained_visits'] == 0


@pytest.mark.parametrize('fen', [
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '7k/8/8/3pP3/8/8/8/7K w - d6 0 1',
    '7k/P7/8/8/8/8/8/7K w - - 0 1',
])
def test_special_rule_inputs_produce_legal_moves(command, fen):
    with chess.engine.SimpleEngine.popen_uci(command, timeout=30) as engine:
        board = chess.Board(fen)
        result = engine.play(board, chess.engine.Limit(nodes=64))
        assert result.move in board.legal_moves


def test_early_stop_does_not_strand_virtual_reservations(command):
    with chess.engine.SimpleEngine.popen_uci(command, timeout=30) as engine:
        board, game = chess.Board(), object()
        analysis = engine.analysis(board, chess.engine.Limit(time=5), game=game)
        time.sleep(.02)
        analysis.stop()
        result = analysis.wait()
        assert result.move in board.legal_moves
        resumed = engine.play(board, chess.engine.Limit(nodes=128), game=game, info=chess.engine.INFO_ALL)
        assert resumed.move in board.legal_moves
        assert telemetry(resumed)['native']['roots'][0]['new_simulations'] == 128


def test_large_search_exceeds_original_node_and_edge_reserves(command):
    with chess.engine.SimpleEngine.popen_uci(command, timeout=30) as engine:
        result = engine.play(chess.Board(), chess.engine.Limit(nodes=8192), info=chess.engine.INFO_ALL)
        data = telemetry(result)
        assert data['native']['roots'][0]['new_simulations'] == 8192
        assert data['native']['roots'][0]['nodes'] > 4096


def test_illegal_history_is_rejected_and_next_position_recovers(command):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1)
    lines = queue.Queue()
    def read():
        for line in process.stdout:
            lines.put(line.strip())
    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    def send(text):
        process.stdin.write(text + '\n')
        process.stdin.flush()
    def until(prefix):
        found = []
        while True:
            line = lines.get(timeout=30)
            found.append(line)
            if line.startswith(prefix):
                return found
    try:
        send('uci'); until('uciok')
        send('position startpos moves e2e4 e7e4')
        send('go nodes 64')
        rows = until('bestmove ')
        assert rows[-1] == 'bestmove 0000'
        assert any('Illegal root history' in line for line in rows)
        send('position startpos moves e2e4')
        send('go nodes 64')
        rows = until('bestmove ')
        board = chess.Board(); board.push_uci('e2e4')
        assert chess.Move.from_uci(rows[-1].split()[1]) in board.legal_moves
    finally:
        send('quit')
        process.wait(timeout=30)
