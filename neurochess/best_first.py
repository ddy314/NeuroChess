"""Native best-first minimax; Python handles only sparse large-model queries."""
from collections import OrderedDict
import ctypes as C
import json
from pathlib import Path
import time

import chess
import numpy as np

from .native_mcts import history


class Config(C.Structure):
    _fields_ = [(key, C.c_int) for key in ('expansions', 'max_nodes', 'depth', 'queries', 'batch', 'qdepth', 'qbudget')]


def load_library():
    root = Path(__file__).with_name('native')
    path = root/'libbestfirst.so'
    if not path.exists() or any(p.stat().st_mtime > path.stat().st_mtime for p in [
            root/'best_first.cpp', root/'chess_rules.hpp', root/'vendor/chess.hpp']):
        raise RuntimeError('Build search first: python scripts/build_best_first.py')
    lib = C.CDLL(str(path)); ptr = C.c_void_p
    lib.bf_error.restype = C.c_char_p
    lib.bf_create.argtypes = [C.c_char_p, Config]; lib.bf_create.restype = ptr
    lib.bf_destroy.argtypes = [ptr]
    lib.bf_ablation.argtypes = [ptr,C.c_int,C.c_int,C.c_int]
    lib.bf_forget.argtypes = [ptr]
    lib.bf_root_coverage.argtypes = [ptr,C.c_int]
    lib.bf_local_exploration.argtypes = [ptr,C.c_int]
    lib.bf_safety.argtypes = [ptr,C.c_float,C.c_int,C.c_int,C.c_int,C.c_int]
    lib.bf_root_clip.argtypes = [ptr,C.c_float]
    lib.bf_cpu_clip.argtypes = [ptr,C.c_float]
    lib.bf_start.argtypes = [ptr, C.c_char_p, C.c_char_p]
    lib.bf_options.argtypes = [ptr, C.c_float, C.c_float, C.c_float, C.c_int, C.c_float, C.c_int, C.c_int, C.c_int]
    lib.bf_next.argtypes = [ptr, C.c_int, ptr, ptr, ptr]
    lib.bf_submit.argtypes = [ptr, ptr, ptr, ptr, C.c_int]
    lib.bf_result.argtypes = [ptr]; lib.bf_result.restype = C.c_char_p
    lib.bf_trace.argtypes = [ptr]; lib.bf_trace.restype = C.c_char_p
    lib.bf_probe.argtypes = [ptr, C.c_char_p, C.c_char_p, ptr, ptr]
    return lib


class BestFirstPlayer:
    mode = 'best-first-minimax'

    def __init__(self, cheap_weights, network=None, expansions=512, max_nodes=65536,
                 max_depth=64, neural_budget=32, correction_batch=4, cache_size=32768,
                 quiescence_depth=0, quiescence_nodes=100000, anchor_weight=.75,
                 q_anchor_weight=.25, anchor_decay=0., correction_scheduler='hierarchical',
                 root_policy_weight=0., align_root_objective=False,
                 verify_final=False, verification_reserve=8, local_exploration=False,
                 correction_clip=0.,mate_nodes=0,mate_depth=7,tactical_nodes=0,tactical_depth=2,
                 root_value_clip=0.,cpu_correction_clip=0.,tree_reuse=False,
                 backup_mode='legacy',cheap_evaluator='student',root_coverage=0):
        if network is None:
            neural_budget = 0
        if cache_size < 0:
            raise ValueError('Nonnegative cache size required')
        self.network = network
        self.device = getattr(network, 'device', 'cpu')
        self.lib = load_library()
        self._config = Config(expansions, max_nodes, max_depth, neural_budget, correction_batch,
                              quiescence_depth, quiescence_nodes)
        self.handle = self.lib.bf_create(str(Path(cheap_weights).resolve()).encode(), self._config)
        if not self.handle:
            raise ValueError(self.lib.bf_error().decode())
        backups={'legacy':0,'visit-mean':1,'evidence-minimax':2,'neural-mean':3}
        evaluators={'student':0,'material':1,'neutral':2}
        if backup_mode not in backups or cheap_evaluator not in evaluators:
            self.close();raise ValueError('Unknown backup or cheap evaluator')
        self._check(self.lib.bf_ablation(self.handle,int(tree_reuse),backups[backup_mode],evaluators[cheap_evaluator]))
        self._check(self.lib.bf_root_coverage(self.handle,root_coverage))
        if correction_scheduler not in ('hierarchical','frontier'):
            self.close()
            raise ValueError('Unknown correction scheduler')
        rc=self.lib.bf_options(self.handle, anchor_weight, q_anchor_weight, anchor_decay,
                              int(correction_scheduler=='frontier'), root_policy_weight,int(align_root_objective),
                              int(verify_final),verification_reserve)
        if rc<0:
            message=self.lib.bf_error().decode();self.close()
            raise ValueError(message)
        self._check(self.lib.bf_local_exploration(self.handle,int(local_exploration)))
        rc=self.lib.bf_safety(self.handle,correction_clip,mate_nodes,mate_depth,tactical_nodes,tactical_depth)
        if rc<0:
            message=self.lib.bf_error().decode();self.close();raise ValueError(message)
        rc=self.lib.bf_root_clip(self.handle,root_value_clip)
        if rc<0:
            message=self.lib.bf_error().decode();self.close();raise ValueError(message)
        rc=self.lib.bf_cpu_clip(self.handle,cpu_correction_clip)
        if rc<0:
            message=self.lib.bf_error().decode();self.close();raise ValueError(message)
        self.options=dict(anchor_weight=anchor_weight,q_anchor_weight=q_anchor_weight,
                          anchor_decay=anchor_decay,correction_scheduler=correction_scheduler,
                          root_policy_weight=root_policy_weight,align_root_objective=bool(align_root_objective),
                          verify_final=bool(verify_final),verification_reserve=verification_reserve,
                          local_exploration=bool(local_exploration),correction_clip=correction_clip,
                          mate_nodes=mate_nodes,mate_depth=mate_depth,tactical_nodes=tactical_nodes,tactical_depth=tactical_depth,
                          root_value_clip=root_value_clip,cpu_correction_clip=cpu_correction_clip,
                          tree_reuse=bool(tree_reuse),backup_mode=backup_mode,cheap_evaluator=cheap_evaluator,
                          root_coverage=root_coverage)
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.tokens = np.empty((correction_batch, 64), np.uint8)
        self.meta = np.empty((correction_batch, 12), np.float32)
        self.ids = np.empty((correction_batch, 256), np.int16)
        self.search_stats = dict(choices=0, cheap_evaluations=0, expanded_nodes=0,
            correction_positions=0, large_model_positions=0, inference_batches=0,
            cache_hits=0, native_seconds=0., inference_seconds=0., max_depth=0,verification_queries=0,
            mate_nodes=0,tactical_nodes=0,cpu_mate_proofs=0,cpu_references=0,clipped_anchor_nodes=0,
            tree_reuses=0,retained_nodes=0)
        self.decision_samples = []

    @property
    def config(self):
        c = self._config
        return dict(algorithm='best-first negamax with lazy multi-fidelity evaluation',
            backend='cpp', expansions_per_move=c.expansions, max_nodes=c.max_nodes,
            max_depth=c.depth, neural_budget=c.queries, correction_batch=c.batch,
            value_backup=self.options['backup_mode']+'; exact outcomes override',
            exact_proofs='all legal children required for loss/draw',
            correction_scheduler_description='candidate calibration ranked by minimax gaps; heuristic',
            **self.options,
            quiescence_depth=c.qdepth, quiescence_extra_node_cap=c.qbudget,
            quiescence='CPU captures/promotions and all check evasions; no large-model queries',
            cache='exact effective model inputs plus ordered legal action IDs; outputs only')

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.bf_destroy(self.handle); self.handle = None

    def __del__(self):
        self.close()

    def _check(self, rc):
        if rc < 0:
            raise RuntimeError(self.lib.bf_error().decode())
        return rc

    def clear_cache(self):
        """Call after changing the network weights or inference configuration."""
        self.cache.clear()
        self._check(self.lib.bf_forget(self.handle))

    def trace(self):
        """Diagnostic tree/correction snapshot; excluded from normal search timing."""
        if not self.handle:
            raise ValueError('Search is closed')
        raw = self.lib.bf_trace(self.handle)
        if raw is None:
            raise RuntimeError(self.lib.bf_error().decode())
        return json.loads(raw)

    def _evaluate(self, n):
        policy = np.empty((n, 256), np.float32)
        q = np.full((n, 256), np.nan, np.float32)
        values = np.empty(n, np.float32)
        missing, keys = [], []
        model = getattr(self.network, 'model', None)
        history_free = getattr(getattr(model, 'config', None), 'history_free', False)
        for i in range(n):
            meta = self.meta[i].copy()
            if history_free:
                meta[[5,6,7,10]] = 0
            key = self.tokens[i].tobytes()+meta.tobytes()+self.ids[i].tobytes()
            keys.append(key)
            if key in self.cache:
                policy[i], q[i], values[i] = self.cache[key]
                self.cache.move_to_end(key)
                self.search_stats['cache_hits'] += 1
            else:
                missing.append(i)
        if missing:
            # Deduplicate equal inputs in this batch too, without merging rule states.
            unique = list(dict.fromkeys(keys[i] for i in missing))
            representatives = [keys.index(k) for k in unique]
            t, m, ids = self.tokens[representatives], self.meta[representatives], self.ids[representatives]
            if hasattr(self.network, 'evaluate_search_heads'):
                p, nq, v = self.network.evaluate_search_heads(t, m, ids)
            else:
                p, v = self.network.evaluate_encoded(t, m, ids)
                nq = np.full((len(t), 256), np.nan, np.float32)
            p, nq, v = np.asarray(p, np.float32), np.asarray(nq, np.float32), np.asarray(v, np.float32)
            if p.shape != (len(t), 256) or nq.shape != p.shape or v.shape != (len(t),):
                raise ValueError('Expected policy/Q (batch,256), value (batch,)')
            if not np.isfinite(p[ids >= 0]).all() or not np.isfinite(v).all() or np.any(np.abs(v)>1.0001):
                raise FloatingPointError('Invalid large model outputs')
            if np.isinf(nq).any() or np.any(np.abs(nq[np.isfinite(nq)])>1.0001):
                raise FloatingPointError('Invalid large model candidate Q')
            rows = {key:(p[j].copy(), nq[j].copy(), float(v[j])) for j,key in enumerate(unique)}
            for i in missing:
                policy[i], q[i], values[i] = rows[keys[i]]
            if self.cache_size:
                self.cache.update(rows)
                while len(self.cache)>self.cache_size:
                    self.cache.popitem(last=False)
            self.search_stats['large_model_positions'] += len(t)
            self.search_stats['inference_batches'] += 1
        return (np.ascontiguousarray(policy), np.ascontiguousarray(q), np.ascontiguousarray(values))

    def choose(self, board):
        if not self.handle:
            raise ValueError('Search is closed')
        start = time.perf_counter()
        self._check(self.lib.bf_start(self.handle, *history(board)))
        self.search_stats['native_seconds'] += time.perf_counter()-start
        while True:
            start = time.perf_counter()
            n = self._check(self.lib.bf_next(self.handle, len(self.tokens),
                self.tokens.ctypes.data, self.meta.ctypes.data, self.ids.ctypes.data))
            self.search_stats['native_seconds'] += time.perf_counter()-start
            if not n:
                break
            start = time.perf_counter(); policy, q, values = self._evaluate(n)
            self.search_stats['inference_seconds'] += time.perf_counter()-start
            start = time.perf_counter()
            self._check(self.lib.bf_submit(self.handle, policy.ctypes.data, q.ctypes.data, values.ctypes.data, n))
            self.search_stats['native_seconds'] += time.perf_counter()-start
        raw = self.lib.bf_result(self.handle)
        if raw is None:
            raise RuntimeError(self.lib.bf_error().decode())
        result = json.loads(raw)
        move = chess.Move.from_uci(result['move'])
        if move not in board.legal_moves:
            raise RuntimeError('Illegal search result')
        self.search_stats['choices'] += 1
        for key in ['cheap_evaluations', 'expanded_nodes', 'correction_positions','verification_queries',
                    'mate_nodes','tactical_nodes','cpu_mate_proofs','cpu_references','clipped_anchor_nodes',
                    'tree_reuses','retained_nodes']:
            self.search_stats[key] += result['stats'][key]
        self.search_stats['max_depth'] = max(self.search_stats['max_depth'], result['stats']['max_depth'])
        if len(self.decision_samples)<32:
            self.decision_samples.append(dict(fen=board.fen(), **result))
        self.last_result = result
        return move

    def choose_many(self, boards):
        if not boards:
            raise ValueError('Nonempty boards required')
        return [self.choose(board) for board in boards]
