"""Background pipeline: optional HTTPS preprocessing -> audit -> training."""
import argparse
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from .telemetry import write_status


def run_job(spec):
    args = spec['args']
    def argument(name):
        return args[args.index(name) + 1]
    out = Path(argument('--out')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_id = hashlib.sha256(str(out).encode()).hexdigest()
    with open(Path(tempfile.gettempdir()) / f'neurochess-pipeline-{lock_id}.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A pipeline is already running for this output directory') from None
        try:
            if not (out / 'run.json').exists():
                (out / 'run.json').write_text(json.dumps(dict(out=str(out), data=argument('--data'),
                    steps=int(argument('--steps')) if '--steps' in args else 100000)))
            dataset = Path(argument('--data'))
            manifest = dataset / 'manifest.json'
            source = spec.get('pgn_url')
            profile_path = spec.get('data_profile', 'configs/data/elite-recent.json')
            profile = json.loads(Path(profile_path).read_text()) if not source else None
            cache = Path(spec['data_cache']) if spec.get('data_cache') else None
            if not manifest.exists() and cache and (cache / 'manifest.json').exists():
                write_status(out, 'copying_data', note='Copying prepared dataset to VM-local storage')
                shutil.copytree(cache, dataset)
            if not manifest.exists():
                command = [sys.executable, '-u', '-m', 'neurochess.prepare', '--out', str(dataset), '--status-dir', str(out)]
                if source:
                    command += ['--pgn', source, '--min-elo', str(spec.get('min_elo',2000)), '--max-positions', str(spec.get('max_positions',1000000))]
                else:
                    command += ['--profile', profile_path]
                subprocess.run(command, check=True)
            metadata = json.loads(manifest.read_text())
            if source:
                expected = dict(source=source, min_elo=spec.get('min_elo', 2000), max_positions=spec.get('max_positions', 1000000))
                if any(metadata['provenance'].get(k) != v for k, v in expected.items()):
                    raise ValueError('Existing dataset differs from requested source/filters; choose a new --data directory')
            elif metadata['provenance'].get('profile') != profile and spec.get('require_profile', False):
                raise ValueError('Existing dataset profile differs from requested profile; choose a new --data directory')
            (out / 'dataset.json').write_text(json.dumps(metadata, indent=2))
            write_status(out, 'auditing')
            subprocess.run([sys.executable, '-u', '-m', 'neurochess.audit', '--data', str(dataset), '--status-dir', str(out)], check=True)
            if cache and not (cache / 'manifest.json').exists():
                write_status(out, 'copying_data', note='Persisting prepared dataset for future runtimes')
                cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(dataset, cache, ignore=shutil.ignore_patterns('positions.sqlite', 'manifest.json'))
                shutil.copy2(manifest, cache / 'manifest.json')
            if spec.get('prepare_only'):
                write_status(out, 'prepared', data=dict(positions=sum(metadata['counts'].values()),
                    counts=metadata['counts'], sources=metadata['provenance'].get('sources', [])))
                return
            if spec.get('arena') and '--arena-every' not in args:
                write_status(out, 'setup_evaluator')
                engine = spec.get('stockfish') or shutil.which('stockfish')
                if not engine and Path('/usr/games/stockfish').exists():
                    engine = '/usr/games/stockfish'
                if not engine:
                    subprocess.run(['apt-get', 'update', '-qq'], check=True)
                    subprocess.run(['apt-get', 'install', '-y', '-qq', 'stockfish'], check=True)
                    engine = '/usr/games/stockfish'
                args += ['--arena-every','1000','--arena-stockfish',engine,
                    '--arena-openings','configs/monitor-openings.fen','--arena-pairs','4',
                    '--arena-nodes','1000','--arena-max-seconds','120']
            subprocess.run([sys.executable, '-u', '-m', 'neurochess.train', *args], check=True)
        except BaseException as error:
            write_status(out, 'failed', note=str(error))
            raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True)
    a = p.parse_args()
    run_job(json.loads(Path(a.spec).read_text()))


if __name__ == '__main__':
    main()
