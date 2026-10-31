"""Stream a version-pinned HTTP file using checked, retryable byte ranges.

Only file bytes go to stdout. Bounded parallel requests avoid a single long
response ending silently before its advertised Content-Length.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def fetch_range(url, etag, start, end):
    for attempt in range(12):
        try:
            request = Request(url, headers={'Range': f'bytes={start}-{end}', 'If-Range': etag})
            with urlopen(request, timeout=45) as response:
                if response.status != 206 or response.headers.get('ETag') != etag:
                    raise RuntimeError('Source version changed or server ignored byte range')
                content_range = response.headers.get('Content-Range', '')
                if not content_range.startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError(f'Unexpected Content-Range: {content_range}')
                data = response.read()
            if len(data) != end - start + 1:
                raise OSError(f'Short HTTP range: {len(data)} bytes')
            return data
        except HTTPError as error:
            if error.code != 429 or attempt == 11:
                raise
            delay = max(60, float(error.headers.get('Retry-After', 60)))
            print(f'Source rate limit; waiting {delay:g}s before retrying range {start}',file=sys.stderr,flush=True)
            time.sleep(delay)
        except (OSError, RuntimeError):
            if attempt == 11: raise
            time.sleep(min(30, 2 ** attempt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--etag', required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--chunk-bytes', type=int, default=16 * 1024 * 1024)
    parser.add_argument('--connections', type=int, default=2)
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    args.etag = '"' + args.etag.strip('"') + '"'
    if min(args.size, args.chunk_bytes, args.connections) <= 0:
        parser.error('positive size and concurrency required')
    for attempt in range(12):
        try:
            with urlopen(Request(args.url, method='HEAD'), timeout=30) as response:
                if response.headers.get('ETag') != args.etag or int(response.headers['Content-Length']) != args.size:
                    raise RuntimeError('Pinned source metadata does not match HEAD')
            break
        except HTTPError as error:
            if error.code != 429 or attempt == 11: raise
            time.sleep(max(60,float(error.headers.get('Retry-After',60))))
    cache = Path(args.cache_dir) if args.cache_dir else None
    if cache:
        cache.mkdir(parents=True,exist_ok=True)
        version = cache/'version.txt'
        value = f'{args.url}\n{args.etag}\n{args.size}\n{args.chunk_bytes}\n'
        if version.exists() and version.read_text() != value: raise RuntimeError('Cache source version differs')
        version.write_text(value)
    def fetch(start,end):
        path=cache/f'{start:016d}.part' if cache else None
        if path and path.exists() and path.stat().st_size==end-start+1:return path.read_bytes()
        data=fetch_range(args.url,args.etag,start,end)
        if path:
            temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
        return data
    pool = ThreadPoolExecutor(args.connections)
    pending = deque()
    ranges = iter(range(0, args.size, args.chunk_bytes))
    def submit():
        start = next(ranges, None)
        if start is not None:
            pending.append(pool.submit(fetch, start,
                                       min(args.size - 1, start + args.chunk_bytes - 1)))
    for _ in range(args.connections * 2):
        submit()
    try:
        while pending:
            sys.stdout.buffer.write(pending.popleft().result())
            sys.stdout.buffer.flush()
            submit()
    except BrokenPipeError:
        # The importer deliberately closes the stream after its verified target.
        sys.stdout = open('/dev/null', 'w')
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
