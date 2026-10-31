"""Thin batch inference bridge to the native search engine (no per-node callbacks)."""
import ctypes as C
import json
import math
from pathlib import Path
import time

import chess
import numpy as np


class Config(C.Structure):
    _fields_ = [(k, C.c_int) for k in ('simulations', 'batch', 'width', 'keep', 'depth')] + [
        (k, C.c_float) for k in ('cpuct', 'fpu', 'widening', 'prune')]


def load_library():
    path = Path(__file__).with_name('native') / 'libmcts.so'
    sources = [path.with_name('mcts.cpp'), path.with_name('chess_rules.hpp'), path.parent / 'vendor/chess.hpp']
    if not path.exists() or any(p.stat().st_mtime > path.stat().st_mtime for p in sources):
        raise RuntimeError('Build native MCTS first: python scripts/build_native_mcts.py')
    lib = C.CDLL(str(path))  # ctypes releases the GIL during native search.
    ptr = C.c_void_p
    lib.nc_error.restype = C.c_char_p
    lib.nc_create.argtypes = [Config]; lib.nc_create.restype = ptr
    lib.nc_destroy.argtypes = [ptr]
    lib.nc_forget.argtypes = [ptr]
    lib.nc_time_progress.argtypes = [ptr,C.c_float]
    lib.nc_batch_safety.argtypes = [ptr,C.c_int]
    lib.nc_start.argtypes = [ptr, C.c_int, C.c_char_p, C.c_char_p]
    lib.nc_resize.argtypes = [ptr, C.c_int]
    lib.nc_next.argtypes = [ptr, C.c_int, ptr, ptr, ptr, ptr]
    lib.nc_submit.argtypes = [ptr, ptr, ptr, C.c_int]
    lib.nc_submit_heads.argtypes = [ptr, ptr, ptr, ptr, C.c_int]
    lib.nc_result.argtypes = [ptr]; lib.nc_result.restype = C.c_char_p
    lib.nc_probe.argtypes = [C.c_char_p, C.c_char_p, ptr, ptr, ptr, ptr, ptr]
    return lib


def history(board):
    if board.chess960 or type(board) is not chess.Board or not board.is_valid() or any(not m for m in board.move_stack):
        raise ValueError('Native prototype requires valid standard chess and legal non-null history')
    return board.root().fen(en_passant='fen').encode(), ' '.join(m.uci() for m in board.move_stack).encode()


class NativeMCTSPlayer:
    mode = 'mcts'

    def __init__(self, network, simulations=256, c_puct=1.5, fpu_reduction=.2,
                 initial_width=6, widening=1.5, root_keep=12, prune_after=.75,
                 max_depth=128, leaf_batch_size=16,tree_reuse=True,
                 time_limit=None,time_margin=.03,inference_cache_size=0,pending_loss_guard=False):
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit<=0):
            raise ValueError('Positive finite time limit required')
        if not math.isfinite(time_margin) or time_margin<0 or (time_limit is not None and time_margin>=time_limit):
            raise ValueError('Invalid time margin')
        self.time_limit,self.time_margin=time_limit,time_margin
        self.time_stops=0
        self.pending_loss_guard=bool(pending_loss_guard)
        if inference_cache_size<0:raise ValueError('Nonnegative inference cache size required')
        self.inference_cache_size=inference_cache_size
        if inference_cache_size:
            from .inference_cache import CachedSearchNetwork
            network=CachedSearchNetwork(network,inference_cache_size)
        self.tree_reuse=bool(tree_reuse)
        self.network, self.device = network, getattr(network, 'device', 'cpu')
        self.lib = load_library()
        self._config = Config(simulations, leaf_batch_size, initial_width, root_keep, max_depth,
                              c_puct, fpu_reduction, widening, prune_after)
        self.handle = self.lib.nc_create(self._config)
        if not self.handle:
            raise ValueError(self.lib.nc_error().decode())
        self._check(self.lib.nc_batch_safety(self.handle,int(self.pending_loss_guard)))
        self.tokens = np.empty((leaf_batch_size, 64), np.uint8)
        self.meta = np.empty((leaf_batch_size, 12), np.float32)
        self.ids = np.empty((leaf_batch_size, 256), np.int16)
        self.counts = np.empty(leaf_batch_size, np.int32)
        self.search_stats, self.decision_samples = {}, []
        self.timing = dict(native_seconds=0., inference_seconds=0., bridge_seconds=0.)

    @property
    def config(self):
        c = self._config
        return dict(algorithm='PUCT', backend='cpp', simulations_per_move=c.simulations,
                    c_puct=c.cpuct, fpu_reduction=c.fpu, initial_width=c.width,
                    progressive_widening=c.widening, root_keep=c.keep,
                    prune_after_fraction=c.prune, max_depth=c.depth, leaf_batch_size=c.batch,
                    tree_reuse=self.tree_reuse, transposition_merging=False, root_noise=False,
                    time_limit_seconds=self.time_limit,time_margin_seconds=self.time_margin,
                    time_control='completed-batch boundaries; reserve observed round latency; one game slot',
                    timed_pruning='elapsed fraction of time budget; fixed simulation fraction otherwise',
                    inference_cache_size=self.inference_cache_size,
                    pending_loss_guard=self.pending_loss_guard,
                    inference_cache='effective model inputs and ordered legal IDs; outputs only, separate rule trees',
                    batch_selection='virtual loss; pending leaves excluded',
                    reuse_identity='initial FEN plus entire legal move history, per batch slot',
                    root_pruning='retain visit leaders, top-4 priors, forcing moves and proven wins',
                    budget='additional simulations per move, retained visits do not consume budget')

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.nc_destroy(self.handle); self.handle = None

    def clear_cache(self):
        if self.inference_cache_size:self.network.clear_cache()
        if self.handle:self._check(self.lib.nc_forget(self.handle))

    def __del__(self):
        self.close()

    def _check(self, rc):
        if rc < 0:
            raise RuntimeError(self.lib.nc_error().decode())
        return rc

    def choose_many(self, boards):
        if not boards or not self.handle:
            raise ValueError('Nonempty boards and an open player required')
        if self.time_limit is not None and len(boards)!=1:
            raise ValueError('Time limit currently supports one game slot only')
        choice_start=time.perf_counter();rounds=0;longest_round=0.;time_stopped=False
        start = time.perf_counter()
        if not self.tree_reuse:self._check(self.lib.nc_forget(self.handle))
        self._check(self.lib.nc_resize(self.handle, len(boards)))
        for slot, board in enumerate(boards):
            fen, moves = history(board)
            self._check(self.lib.nc_start(self.handle, slot, fen, moves))
        self.timing['bridge_seconds'] += time.perf_counter() - start
        self._check(self.lib.nc_time_progress(self.handle,-1 if self.time_limit is None else 0))
        while True:
            # Stop only at a completed batch boundary: no pending inference or
            # virtual losses are left behind. Always establish the root first.
            if self.time_limit is not None and rounds and time.perf_counter()-choice_start>=self.time_limit-max(self.time_margin,1.15*longest_round):
                time_stopped=True;self.time_stops+=1;break
            start = time.perf_counter()
            round_start=start
            if self.time_limit is not None:
                self._check(self.lib.nc_time_progress(self.handle,min(1.,(start-choice_start)/self.time_limit)))
            n = self._check(self.lib.nc_next(self.handle, len(self.tokens),
                self.tokens.ctypes.data, self.meta.ctypes.data, self.ids.ctypes.data, self.counts.ctypes.data))
            self.timing['native_seconds'] += time.perf_counter() - start
            if not n:
                break
            start = time.perf_counter()
            heads = hasattr(self.network, 'evaluate_search_heads')
            if heads:
                policy, neural_q, values = self.network.evaluate_search_heads(self.tokens[:n], self.meta[:n], self.ids[:n])
                neural_q = np.ascontiguousarray(neural_q, dtype=np.float32)
                if neural_q.shape != (n, 256): raise ValueError('Expected candidate Q (batch,256)')
            else:
                policy, values = self.network.evaluate_encoded(self.tokens[:n], self.meta[:n], self.ids[:n])
            policy = np.ascontiguousarray(policy, dtype=np.float32)
            values = np.ascontiguousarray(values, dtype=np.float32)
            if policy.shape != (n, 256) or values.shape != (n,):
                raise ValueError('Expected encoded policy (batch,256) and values (batch,)')
            self.timing['inference_seconds'] += time.perf_counter() - start
            start = time.perf_counter()
            if heads:
                self._check(self.lib.nc_submit_heads(self.handle, policy.ctypes.data, neural_q.ctypes.data, values.ctypes.data, n))
            else:
                self._check(self.lib.nc_submit(self.handle, policy.ctypes.data, values.ctypes.data, n))
            self.timing['native_seconds'] += time.perf_counter() - start
            rounds+=1;longest_round=max(longest_round,time.perf_counter()-round_start)
        result = json.loads(self.lib.nc_result(self.handle))
        self.search_stats = result['stats'] | self.timing
        self.search_stats['time_stops']=self.time_stops
        if self.inference_cache_size:self.search_stats['inference_cache']=dict(self.network.search_stats)
        elapsed=time.perf_counter()-choice_start
        self.last_result=dict(result['roots'][0],elapsed_seconds=elapsed,time_stopped=time_stopped,
                              time_limit=self.time_limit,time_overrun=self.time_limit is not None and elapsed>self.time_limit)
        for board, root in zip(boards, result['roots']):
            if len(self.decision_samples) < 8:
                self.decision_samples.append(dict(fen=board.fen(), **root))
        moves = [chess.Move.from_uci(r['move']) for r in result['roots']]
        if any(m not in b.legal_moves for m, b in zip(moves, boards)):
            raise RuntimeError('Native search returned illegal move')
        return moves

    def choose(self, board):
        return self.choose_many([board])[0]
