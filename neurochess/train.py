import argparse
import atexit
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import Batches, ReplayBatches, MixtureBatches
from .model import ChessModel, ModelConfig
from .telemetry import append_json, write_status


def losses(outputs, batch, q_weight=1., value_weight=.5):
    policy, q, value = outputs
    target = batch['policy'].float()
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    pl = -(target * F.log_softmax(policy, -1)).sum(-1).mean()
    mask = batch['qmask']
    ql = ((q - batch['q'].float()).square() * mask).sum() / mask.sum().clamp_min(1)
    vl = (-(batch['value'] * F.log_softmax(value, -1)).sum(-1) * batch['value_weight']).sum() / batch['value_weight'].sum().clamp_min(1)
    return pl + q_weight * ql + value_weight * vl, {'policy': pl.detach(), 'q': ql.detach(), 'value': vl.detach()}


def load_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('architecture')=='policy_value_pair_v1':
        from .policy_value_pair import PolicyValuePair
        model=PolicyValuePair(ModelConfig(**ckpt['model_config'])).to(device)
        model.set_value_training(ckpt.get('value_training_mode','head'))
    else:
        model = ChessModel(ModelConfig(**ckpt['model_config'])).to(device)
    model.load_state_dict(ckpt['model'])
    return model, ckpt


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config', default='configs/base.json')
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=0, help='0 keeps PyTorch default; otherwise limits CPU operator threads')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--eval-batches', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int, help='Graceful pilot stop at this absolute step; scheduler still uses --steps')
    p.add_argument('--init', help='Weights-only initialization for a new stage')
    p.add_argument('--continue-from', help='New data stage retaining model and optimizer; resets data cursor and schedule')
    p.add_argument('--validation-data', help='Fixed reference dataset for validation; defaults to --data')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--arena-every', type=int, default=0, help='Periodic match interval in optimizer steps; 0 disables')
    p.add_argument('--arena-stockfish')
    p.add_argument('--arena-openings')
    p.add_argument('--arena-pairs', type=int, default=4)
    p.add_argument('--arena-nodes', type=int, default=1000)
    p.add_argument('--arena-max-plies', type=int, default=200)
    p.add_argument('--arena-max-seconds', type=float, default=120)
    p.add_argument('--q-weight', type=float, default=1.)
    p.add_argument('--value-weight', type=float, default=.5)
    p.add_argument('--q-warmup', type=int, default=0)
    p.add_argument('--history-free', action='store_true')
    p.add_argument('--replay-data')
    p.add_argument('--hard-data',help='Use 70/20/10 core/hard/human microbatch mixture')
    p.add_argument('--mixture-core-batches',type=int,default=7)
    p.add_argument('--mixture-hard-batches',type=int,default=2)
    p.add_argument('--book-data',help='Optional policy-only curated opening replay shared by both experiment arms')
    p.add_argument('--book-every',type=int,default=10)
    p.add_argument('--replay-every', type=int, default=8)
    a = p.parse_args()
    if min(a.mixture_core_batches,a.mixture_hard_batches,a.book_every)<1:
        p.error('Positive mixture counts required')
    if a.book_data and not a.hard_data:p.error('Opening replay requires the hard-data mixture')
    if min(a.q_weight,a.value_weight,a.q_warmup) < 0 or a.replay_every < 2:
        p.error('invalid loss/replay settings')
    if a.log_every < 1 or a.arena_every < 0:
        p.error('invalid logging or arena interval')
    arena_fens = []
    if a.arena_every:
        if not a.arena_stockfish or not a.arena_openings:
            p.error('arena requires --arena-stockfish and --arena-openings')
        if min(a.arena_pairs, a.arena_nodes, a.arena_max_plies, a.arena_max_seconds) <= 0:
            p.error('positive arena limits required')
        import chess
        import chess.engine
        arena_fens = [s.strip() for s in Path(a.arena_openings).read_text().splitlines() if s.strip()][:a.arena_pairs]
        if len(arena_fens) != a.arena_pairs:
            p.error('not enough arena opening pairs')
        for fen in arena_fens:
            board = chess.Board(fen)
            if not board.is_valid() or board.is_game_over(claim_draw=True):
                p.error('arena opening is invalid or terminal')
        with chess.engine.SimpleEngine.popen_uci(a.arena_stockfish, timeout=10):
            pass
    if min(a.steps, a.batch_size, a.accum, a.save_every, a.eval_batches) < 1 or a.workers < 0:
        p.error('invalid training limits')
    if sum(bool(x) for x in [a.resume, a.init, a.continue_from]) > 1:
        p.error('resume, init and continue-from are mutually exclusive')
    if a.cpu_threads < 0:
        p.error('cpu-threads must be nonnegative')
    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cuda = a.device.startswith('cuda')
    torch.set_float32_matmul_precision('high')
    dtype = torch.bfloat16 if cuda and (torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
    amp = lambda: torch.autocast('cuda', dtype=dtype) if cuda else nullcontext()
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and dtype == torch.float16)
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    import fcntl
    import tempfile
    lock_id = hashlib.sha256(str(root.resolve()).encode()).hexdigest()
    run_lock = open(Path(tempfile.gettempdir()) / f'neurochess-{lock_id}.lock', 'w')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another training process is using this output directory') from None
    if (root / 'last.pt').exists() and not a.resume:
        raise ValueError('Output already contains a checkpoint; resume or use a new directory')
    Batches(a.data, 'train', a.batch_size)
    Batches(a.validation_data or a.data, 'val', a.batch_size)
    manifest = (Path(a.data) / 'manifest.json').read_bytes()
    if a.validation_data:
        manifest += (Path(a.validation_data) / 'manifest.json').read_bytes()
    if a.replay_data:
        replay_files = sorted(Path(a.replay_data).glob('train-*.npy'))
        if len(replay_files) < max(1,a.workers):
            p.error('replay requires at least one shard per loader worker')
        manifest += (Path(a.replay_data) / 'manifest.json').read_bytes()
    if a.hard_data:
        if not a.replay_data or len(list(Path(a.hard_data).glob('train-*.npy')))<max(1,a.workers):
            p.error('Hard mixture requires human replay and one hard shard per worker')
        manifest += (Path(a.hard_data)/'manifest.json').read_bytes()
    if a.book_data:
        if len(list(Path(a.book_data).glob('train-*.npy')))<max(1,a.workers):
            p.error('Opening replay requires one shard per loader worker')
        manifest += (Path(a.book_data)/'manifest.json').read_bytes()
    if a.hard_data and (a.mixture_core_batches,a.mixture_hard_batches,a.book_every)!=(7,2,10):
        manifest += json.dumps(dict(core=a.mixture_core_batches,hard=a.mixture_hard_batches,book_every=a.book_every),sort_keys=True).encode()
    data_hash = hashlib.sha256(manifest).hexdigest()
    state = None
    if a.resume or a.init or a.continue_from:
        model, state = load_model(a.resume or a.init or a.continue_from, a.device)
    else:
        model = ChessModel(ModelConfig(**json.loads(Path(a.config).read_text()))).to(a.device)
    if a.history_free:
        model.config.history_free = True
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.1, fused=cuda)
    step = epoch = cursor = 0
    origin_step = 0
    lineage = {}
    best_val_loss = float('inf')
    if a.continue_from:
        if 'optimizer' not in state or 'scaler' not in state:
            raise ValueError('continue-from requires a full optimizer checkpoint; use --init for weights only')
        if state['model_config'].get('history_free', False) != a.history_free:
            raise ValueError('continue-from requires unchanged history-free setting')
        opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        origin_step = state.get('origin_step', 0) + state['step']
        lineage = dict(parent_checkpoint=str(Path(a.continue_from).resolve()),
                       parent_step=state['step'], parent_data_hash=state.get('data_hash'),
                       origin_step=origin_step, optimizer_retained=True,
                       data_cursor_reset=True, schedule_reset=True)
    if a.resume:
        if state['data_hash'] != data_hash:
            raise ValueError('resume dataset manifest differs; use --continue-from or --init for a new stage')
        for key in ['batch_size', 'accum', 'workers', 'seed', 'lr', 'warmup', 'steps']:
            if state['args'][key] != getattr(a, key):
                raise ValueError(f'resume requires unchanged {key}')
        defaults = dict(q_weight=1., value_weight=.5, q_warmup=0, history_free=False, replay_data=None, replay_every=8, validation_data=None, hard_data=None,
                        mixture_core_batches=7,mixture_hard_batches=2,book_data=None,book_every=10)
        for key, default in defaults.items():
            if state['args'].get(key,default) != getattr(a,key):
                raise ValueError(f'resume requires unchanged {key}')
        opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        step, epoch, cursor = state['step'], state['epoch'], state['cursor']
        origin_step = state.get('origin_step', 0)
        lineage = state.get('lineage', {})
        best_val_loss = state.get('best_val_loss',float('inf'))
        torch.set_rng_state(state['rng'])
        if cuda and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    runner = torch.compile(model) if a.compile else model
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()), device=a.device,
        precision=str(dtype) if cuda else 'float32', config=asdict(model.config))), flush=True)
    (root / 'run.json').write_text(json.dumps(vars(a), indent=2))
    (root / 'dataset.json').write_text((Path(a.data) / 'manifest.json').read_text())
    (root / 'lineage.json').write_text(json.dumps(lineage, indent=2))
    if a.validation_data:
        (root / 'validation-dataset.json').write_text((Path(a.validation_data) / 'manifest.json').read_text())
    if a.replay_data:
        (root / 'replay-dataset.json').write_text((Path(a.replay_data) / 'manifest.json').read_text())
    if a.hard_data:
        (root/'hard-dataset.json').write_text((Path(a.hard_data)/'manifest.json').read_text())
    if a.book_data:
        (root/'book-dataset.json').write_text((Path(a.book_data)/'manifest.json').read_text())
    (root / 'runtime.json').write_text(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
        device=torch.cuda.get_device_name() if cuda else a.device, precision=str(dtype) if cuda else 'float32',
        gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else None,
        started_at=time.time(), start_step=step, origin_step=origin_step, cpu_threads=torch.get_num_threads()), indent=2))
    finished = [False]
    def on_exit():
        if not finished[0]:
            write_status(root, 'stopped', step=step, total_steps=a.steps, note='Training exited before planned stop; inspect console log')
    atexit.register(on_exit)
    write_status(root, 'training', step=step, total_steps=a.steps)

    def loader(ep, split='train'):
        ds = Batches(a.validation_data if split == 'val' and a.validation_data else a.data,
                     split, a.batch_size, a.seed, ep)
        if split == 'train' and a.hard_data:
            ds = MixtureBatches(ds,a.hard_data,a.replay_data,a.mixture_core_batches,a.mixture_hard_batches,a.book_data,a.book_every)
        elif split == 'train' and a.replay_data:
            ds = ReplayBatches(ds, a.replay_data, a.replay_every)
        return DataLoader(ds, batch_size=None, num_workers=a.workers, pin_memory=cuda)

    def transfer(batch):
        return {k: v.to(a.device, non_blocking=True) for k, v in batch.items()}

    def validate(report_step):
        write_status(root, 'validating', step=report_step, total_steps=a.steps)
        runner.eval()
        totals = torch.zeros(8, device=a.device)
        with torch.inference_mode():
            for idx, batch in enumerate(loader(0, 'val')):
                if idx >= a.eval_batches:
                    break
                batch = transfer(batch)
                with amp():
                    output = runner(batch['tokens'], batch['meta'], batch['moves'])
                    loss, components = losses(output, batch, a.q_weight, a.value_weight)
                n = len(batch['tokens'])
                totals += torch.stack([loss * n, (output[0].argmax(-1) == batch['policy'].argmax(-1)).sum(), torch.tensor(n, device=a.device),
                    *[x*n for x in components.values()],
                    ((output[1]-batch['q'].float()).abs()*batch['qmask']).sum(), batch['qmask'].sum()])
        v = totals.tolist()
        report = dict(step=report_step, cumulative_step=origin_step+report_step,
                      val_loss=v[0] / v[2], val_top1=v[1] / v[2], positions=v[2],
                      val_policy_loss=v[3]/v[2], val_q_loss=v[4]/v[2],
                      val_value_loss=v[5]/v[2], val_q_mae=v[6]/max(1,v[7]), q_labels=v[7])
        print(json.dumps(report), flush=True)
        append_json(root / 'validation.jsonl', dict(report, timestamp=time.time()))
        return report

    if a.continue_from:
        report = validate(0)
        best_val_loss = report['val_loss']
        atomic_save(dict(model=model.state_dict(), model_config=asdict(model.config), step=0,
                         origin_step=origin_step, lineage=lineage, validation=report), root / 'best.pt')
        write_status(root, 'training', step=0, total_steps=a.steps)

    stream = iter(loader(epoch))
    for _ in range(cursor):
        next(stream)
    start = time.perf_counter(); seen = 0
    end_step = min(a.steps, a.stop_after) if a.stop_after is not None else a.steps
    while step < end_step:
        runner.train(); opt.zero_grad(set_to_none=True)
        progress = max(0, step - a.warmup) / max(1, a.steps - a.warmup)
        lr = a.lr * min(1., (step + 1) / max(1, a.warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in opt.param_groups:
            group['lr'] = lr
        metrics = torch.zeros(4, device=a.device)
        q_labels = torch.zeros((), device=a.device); step_seen = 0
        for _ in range(a.accum):
            try:
                batch = next(stream)
            except StopIteration:
                epoch += 1; cursor = 0; stream = iter(loader(epoch)); batch = next(stream)
            cursor += 1
            batch = transfer(batch)
            with amp():
                loss, parts = losses(runner(batch['tokens'], batch['meta'], batch['moves']), batch,
                    a.q_weight * min(1., (step+1)/max(1,a.q_warmup)), a.value_weight)
            scaler.scale(loss / a.accum).backward()
            metrics += torch.stack([loss.detach(), *parts.values()]) / a.accum
            seen += len(batch['tokens']); step_seen += len(batch['tokens']); q_labels += batch['qmask'].sum()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm) and not scaler.is_enabled():
            raise FloatingPointError('Nonfinite gradient')
        scaler.step(opt); scaler.update(); step += 1
        if step % a.log_every == 0 or step == 1 or step == end_step:
            row = dict(step=step, cumulative_step=origin_step+step, epoch=epoch, lr=lr, loss=metrics.tolist(), timestamp=time.time(),
                       positions_per_second=seen / (time.perf_counter() - start),
                       gradient_norm=float(norm) if torch.isfinite(norm) else None,
                       optimizer_step_skipped=bool(scaler.is_enabled() and not torch.isfinite(norm)), q_labels=int(q_labels), batch_positions=step_seen,
                       session_positions=seen, session_seconds=time.perf_counter()-start,
                       q_weight=a.q_weight*min(1.,step/max(1,a.q_warmup)), value_weight=a.value_weight,
                       gpu_allocated_gib=torch.cuda.memory_allocated()/2**30 if cuda else None,
                       gpu_reserved_gib=torch.cuda.memory_reserved()/2**30 if cuda else None,
                       gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None,
                       scaler_scale=scaler.get_scale() if scaler.is_enabled() else None)
            print(json.dumps(row), flush=True)
            append_json(root / 'metrics.jsonl', row)
            write_status(root, 'training', step=step, total_steps=a.steps)
        arena_due = bool(a.arena_every and (step % a.arena_every == 0 or step == end_step))
        if step % a.save_every == 0 or step == end_step or arena_due:
            report = validate(step)
            write_status(root, 'saving', step=step, total_steps=a.steps)
            if report['val_loss'] < best_val_loss:
                best_val_loss = report['val_loss']
                atomic_save(dict(model=model.state_dict(),model_config=asdict(model.config),step=step,origin_step=origin_step,lineage=lineage,
                    validation=report),root / 'best.pt')
            atomic_save(dict(best_val_loss=best_val_loss, model=model.state_dict(), model_config=asdict(model.config), optimizer=opt.state_dict(),
                scaler=scaler.state_dict(), step=step, origin_step=origin_step, lineage=lineage, epoch=epoch, cursor=cursor, args=vars(a), data_hash=data_hash,
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cuda else None), root / 'last.pt')
        if arena_due:
            from .play import Player, run_match
            import uuid
            match_name = f'step-{step:08d}-{uuid.uuid4().hex[:8]}'
            match_out = root / 'arena' / match_name
            write_status(root, 'evaluating', step=step, total_steps=a.steps, games_done=0, games_total=a.arena_pairs * 2)
            try:
                report = run_match(Player.from_model(model, a.device), a.arena_stockfish, arena_fens, match_out,
                    a.arena_nodes, a.arena_max_plies, a.arena_max_seconds,
                    progress=lambda done, total: write_status(root, 'evaluating', step=step,
                        total_steps=a.steps, games_done=done, games_total=total))
                append_json(root / 'arena.jsonl', dict(report, step=step, timestamp=time.time(),
                    pgn=f'arena/{match_name}/games.pgn'))
            except Exception as error:
                failure = dict(step=step, timestamp=time.time(), error=str(error))
                append_json(root / 'arena.jsonl', failure)
                print(json.dumps(dict(arena_error=failure)), flush=True)
            finally:
                runner.train()
                write_status(root, 'training', step=step, total_steps=a.steps)
    finished[0] = True
    write_status(root, 'complete' if step >= a.steps else 'paused', step=step, total_steps=a.steps)



if __name__ == '__main__':
    main()
