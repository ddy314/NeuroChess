"""Release this Colab runtime only after verified successful completion."""
import argparse
import fcntl
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import time

from .persistence import RunMirror


SUCCESS_LINE = 'Expanded training completed and persisted'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process_identity(pid):
    """PID plus kernel start ticks; reject zombies and PID reuse."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if stat[0] == 'Z' else stat[19]
    except (OSError, IndexError):
        return None


def find_runtime_address():
    address = os.environ.get('TBE_RUNTIME_ADDR')
    if address:
        return address
    # SSH children do not inherit the notebook kernel's runtime-service variable.
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if b'colab_kernel_launcher' not in (process/'cmdline').read_bytes():
                continue
            for variable in (process/'environ').read_bytes().split(b'\0'):
                if variable.startswith(b'TBE_RUNTIME_ADDR='):
                    return variable.split(b'=', 1)[1].decode()
        except (OSError, ValueError):
            continue
    raise RuntimeError('Colab runtime-management service is unavailable')


def request_unassign(address):
    # This is the same runtime-local POST used by google.colab.runtime.unassign().
    # Its browser-disconnect JavaScript is unnecessary in this headless process.
    host, port = address.rsplit(':', 1)
    if not ipaddress.ip_address(host).is_private or not 0 < int(port) < 65536:
        raise ValueError('Expected a private runtime-management address')
    connection = http.client.HTTPConnection(host, int(port), timeout=30)
    try:
        connection.request('POST', '/unassign', body=b'')
        response = connection.getresponse()
        response.read()
        if response.status != 200:
            raise RuntimeError(f'Runtime release returned HTTP {response.status}')
    finally:
        connection.close()


def write_state(source, destination, phase, **fields):
    record = dict(phase=phase, updated_at=time.time(), **fields)
    for root in [Path(source), Path(destination)]:
        if not root.is_dir():
            raise OSError(f'Existing run directory is unavailable: {root}')
        path = root/'autostop.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(path)
    return record


def verify_completed_run(source, destination, mount, steps, data_hash):
    import torch
    source, destination, mount = Path(source), Path(destination), Path(mount)
    if not os.path.ismount(mount) or not destination.is_dir():
        raise OSError('Original Drive mount or run directory is unavailable')
    for root in [source, destination]:
        status = json.loads((root/'status.json').read_text())
        config = json.loads((root/'run.json').read_text())
        if status.get('phase') != 'complete' or status.get('step') != steps or status.get('total_steps') != steps or config.get('steps') != steps:
            raise RuntimeError('Training has not completed the expected plan')
    files = [file for file in source.rglob('*') if file.is_file()
             and file.suffix != '.tmp' and not file.name.startswith('autostop')]
    required = {'last.pt', 'best.pt', 'metrics.jsonl', 'validation.jsonl', 'dataset.json',
                'validation-dataset.json', 'replay-dataset.json', 'lineage.json',
                'run.json', 'training-code.tar.gz', 'data-cache.json'}
    if not required.issubset({str(f.relative_to(source)) for f in files}):
        raise RuntimeError('Final run is missing recovery artifacts')
    hashes = {}
    for file in sorted(files):
        relative = str(file.relative_to(source))
        expected = digest(file)
        if digest(destination/relative) != expected:
            raise RuntimeError(f'Drive readback differs: {relative}')
        hashes[relative] = expected
    checkpoint = torch.load(destination/'last.pt', map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('step') != steps or checkpoint.get('data_hash') != data_hash or 'optimizer' not in checkpoint:
        raise RuntimeError('Final persistent checkpoint has wrong step, data fingerprint, or no optimizer')
    cumulative_step = checkpoint.get('origin_step', 0) + steps
    del checkpoint
    cache = json.loads((destination/'data-cache.json').read_text())
    cache_path = Path(cache['path']).resolve()
    if not cache_path.is_relative_to(mount.resolve()) or digest(cache_path) != cache['sha256']:
        raise RuntimeError('Persistent training-data cache failed verification')
    evaluation = None
    if (source/'evaluation-required.json').exists():
        from .evaluation_check import verify_arena
        requirement=json.loads((source/'evaluation-required.json').read_text())
        evaluation=verify_arena(destination/requirement['directory'],requirement['games_per_mode'],destination/'selected.pt')
    return dict(step=steps, cumulative_step=cumulative_step, data_hash=data_hash,
                verified_files=hashes, data_cache_sha256=cache['sha256'], evaluation=evaluation, verified_at=time.time())


def finalize_and_stop(source, destination, mount, steps, data_hash, address=None):
    if not os.path.ismount(mount) or not Path(destination).is_dir():
        raise OSError('Original Drive mount or run directory is unavailable')
    address = address or find_runtime_address()
    write_state(source, destination, 'verifying', expected_steps=steps)
    # Repair any delayed final copies and persist the latest monitoring metadata.
    RunMirror(source, destination, mount).sync()
    verification = verify_completed_run(source, destination, mount, steps, data_hash)
    for root in [Path(source), Path(destination)]:
        path = root/'autostop-verification.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(verification, indent=2)); temporary.replace(path)
    write_state(source, destination, 'shutdown_requested', expected_steps=steps,
                verified_at=verification['verified_at'], checkpoint_sha256=verification['verified_files']['last.pt'])
    # The VM may disappear before this call returns; record intent on Drive first.
    request_unassign(address)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--mount', required=True)
    parser.add_argument('--launcher-pid', type=int, required=True)
    parser.add_argument('--launcher-log', required=True)
    parser.add_argument('--expected-steps', type=int, required=True)
    parser.add_argument('--data-hash', required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.expected_steps < 1:
        parser.error('positive training plan and poll interval required')
    source, destination = Path(args.source), Path(args.destination)
    if not os.path.ismount(args.mount) or not destination.is_dir():
        raise OSError('Original Drive mount or run directory is unavailable')
    lock = (source/'autostop.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = process_identity(args.launcher_pid)
    if identity is None:
        raise RuntimeError('Expected active launcher is not running; auto-stop was not armed')
    command = Path(f'/proc/{args.launcher_pid}/cmdline').read_bytes()
    if b'expanded_colab.py' not in command:
        raise RuntimeError('Launcher PID belongs to an unexpected process')
    address = find_runtime_address()
    record = write_state(source, destination, 'armed', expected_steps=args.expected_steps,
                         data_hash=args.data_hash, launcher_pid=args.launcher_pid,
                         launcher_start_ticks=identity, guard_pid=os.getpid(),
                         poll_seconds=args.poll_seconds, host_independent=True)
    print(json.dumps(record), flush=True)
    while process_identity(args.launcher_pid) == identity:
        time.sleep(args.poll_seconds)
    try:
        lines = Path(args.launcher_log).read_text().splitlines()
        if not lines or lines[-1] != SUCCESS_LINE:
            raise RuntimeError('Launcher exited without its successful final-persistence marker')
        verification = finalize_and_stop(source, destination, args.mount, args.expected_steps,
                                         args.data_hash, address)
        print(json.dumps(dict(unassign_accepted=True, step=verification['step'])), flush=True)
    except Exception as error:
        write_state(source, destination, 'blocked', expected_steps=args.expected_steps, error=str(error))
        raise


if __name__ == '__main__':
    main()
