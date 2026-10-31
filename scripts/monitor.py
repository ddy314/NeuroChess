#!/usr/bin/env python3
"""Open the dashboard of an existing Colab training session."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

COLAB = shutil.which('colab') or str(Path.home() / '.local/bin/colab')


class InactiveSession(RuntimeError):
    pass


def active_endpoint(session):
    result = subprocess.run([COLAB, 'sessions'], capture_output=True, text=True,
                            check=True, timeout=45)
    match = re.search(r'^\[' + re.escape(session) + r'\]\s+(\S+)',
                      result.stdout, re.MULTILINE)
    if not match:
        raise InactiveSession(f'Colab session {session!r} is not active; monitoring does not start training.')
    return match[1]


def read_dashboard(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/runs', timeout=3) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='neurochess-train')
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--no-open', action='store_true', help='Do not open a browser')
    parser.add_argument('--tunnel', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not re.fullmatch(r'[A-Za-z0-9_-]+', args.session):
        parser.error('Use a valid port (1024-65535) and a simple session name')
    try:
        endpoint = active_endpoint(args.session)
    except Exception as error:
        print(error, file=sys.stderr)
        return 0 if args.tunnel and isinstance(error, InactiveSession) else 1
    state = Path.home() / '.local/state/neurochess-monitor'
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = state / 'id_ed25519'
    if not key.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(key)], check=True)
    control = state / f'{args.session}-{args.port}.sock'
    if args.tunnel:
        proxy = ' '.join(shlex.quote(s) for s in
                         [COLAB, 'ssh', '--proxy-mode', '-s', args.session,
                          '-i', str(key)])
        command = ['ssh', '-M', '-S', str(control), '-N',
                   '-L', f'127.0.0.1:{args.port}:127.0.0.1:8765', '-i', str(key),
                   '-o', 'IdentitiesOnly=yes', '-o', f'HostKeyAlias={endpoint}',
                   '-o', 'StrictHostKeyChecking=accept-new',
                   '-o', f'UserKnownHostsFile={state / "known_hosts"}',
                   '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30',
                   '-o', 'ServerAliveCountMax=3', '-o', 'ExitOnForwardFailure=yes',
                   '-o', f'ProxyCommand={proxy}', 'root@colab']
        os.execvp(command[0], command)
    unit = f'neurochess-monitor-{args.session}-{args.port}.service'
    state_result = subprocess.run(['systemctl', '--user', 'show', unit,
                                   '--property=ActiveState', '--value'],
                                  capture_output=True, text=True)
    unit_state = state_result.stdout.strip()
    if unit_state in ('active', 'activating', 'reloading'):
        print(f'Using existing monitor service ({unit_state}).', flush=True)
    else:
        # A failed transient unit can remain loaded until it is stopped and collected.
        subprocess.run(['systemctl', '--user', 'stop', unit],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['systemctl', '--user', 'reset-failed', unit],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start = subprocess.run(['systemd-run', '--user', '--unit', unit, '--collect',
                                '--property=Restart=on-failure', '--property=RestartSec=10',
                                sys.executable, str(Path(__file__).resolve()), '--tunnel',
                                '--session', args.session, '--port', str(args.port)],
                               capture_output=True, text=True)
        if start.returncode:
            # Another invocation may have created the unit at the same time.
            current = subprocess.run(['systemctl', '--user', 'show', unit,
                                      '--property=ActiveState', '--value'],
                                     capture_output=True, text=True).stdout.strip()
            if current not in ('active', 'activating', 'reloading'):
                print(f'Could not start monitor service: {start.stderr.strip()}', file=sys.stderr)
                return 1
    print(f'Connecting to {args.session} ({endpoint})...', flush=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            read_dashboard(args.port)
            break
        except Exception:
            time.sleep(2)
    else:
        print(f'Dashboard is not reachable; the service will keep retrying. '
              f'Inspect: journalctl --user -u {unit} -n 30', file=sys.stderr)
        return 1
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Dashboard ready: {url}')
    if not args.no_open:
        subprocess.Popen(['xdg-open', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
