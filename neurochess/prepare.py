"""Stream local/HTTPS PGN into mmap shards, with reproducible source profiles."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sqlite3
import time
from urllib.request import urlopen
import chess.pgn
import zstandard
from .data import ShardWriter, record
from .encoding import split_for, position_key
from .telemetry import write_status


class CountingReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.count = raw, 0
    def readable(self):
        return True
    def readinto(self, buffer):
        data = self.raw.read(len(buffer)); buffer[:len(data)] = data
        self.count += len(data)
        return len(data)
    def close(self):
        self.raw.close(); super().close()


def base_seconds(value):
    """First period's base time; handles '40/7200:3600+30' conservatively."""
    try:
        first = value.split(':')[0].split('+')[0].split('/')[-1]
        return int(first)
    except (ValueError, AttributeError):
        return -1


def rejection(game, source):
    if game.errors:
        return 'parse_error'
    if game.headers.get('Variant', 'Standard') != 'Standard':
        return 'variant'
    if game.headers.get('Result') not in ['1-0', '0-1', '1/2-1/2']:
        return 'unfinished'
    try:
        rating = min(int(game.headers.get(k, '0')) for k in ['WhiteElo', 'BlackElo'])
    except ValueError:
        return 'missing_rating'
    if rating < source['min_elo']:
        return 'low_or_missing_rating'
    minimum = source.get('min_base_seconds', 0)
    seconds = base_seconds(game.headers.get('TimeControl'))
    if minimum and seconds < minimum and not (seconds < 0 and source.get('allow_unknown_time_control', False)):
        return 'short_or_missing_time_control'
    if any(game.headers.get(k) == 'BOT' for k in ['WhiteTitle', 'BlackTitle']):
        return 'bot'
    if not game.board().is_valid():
        return 'invalid_board'
    return None


class FilteredBuilder(chess.pgn.GameBuilder):
    def __init__(self, source):
        super().__init__()
        self.source = source
    def end_headers(self):
        # Reject by metadata before constructing and validating every move.
        if rejection(self.game, self.source):
            return chess.pgn.SKIP
        return None


def prepare(sources, out, stride=4, status_dir=None, profile=None, deduplicate=False):
    writer = ShardWriter(out)
    status_root = Path(status_dir) if status_dir else None
    if status_root:
        status_root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(Path(out) / 'positions.sqlite') if deduplicate else None
    if db:
        db.execute('CREATE TABLE positions (key BLOB PRIMARY KEY) WITHOUT ROWID')
    count = total_games = rejected = compressed = 0
    last_update = 0.
    summaries = []
    target = sum(s.get('max_positions', 0) for s in sources) if all(s.get('max_positions') for s in sources) else 0
    current, counter = {}, None
    def progress(force=False, phase='preparing', error=None):
        nonlocal last_update
        now = time.time()
        if not force and now - last_update < 2:
            return
        last_update = now
        info = dict(source=current.get('url'), source_name=current.get('name'), source_index=min(len(summaries)+1,len(sources)),
            total_sources=len(sources), positions=count, games=total_games, rejected=rejected,
            compressed_bytes_read=compressed + (counter.count if counter else 0), target_positions=target,
            min_elo=current.get('min_elo'), stride=stride, counts=writer.counts, sources=summaries, error=error)
        if status_root:
            write_status(status_root, phase, data=info)
        if force:
            print(json.dumps(info), flush=True)
    try:
        for source in sources:
            current = source; counter = None
            progress(force=True)
            uri = source['url']
            raw = urlopen(uri, timeout=60) if uri.startswith('https://') else open(uri, 'rb')
            counter = CountingReader(raw)
            reader = zstandard.ZstdDecompressor().stream_reader(counter) if uri.split('?')[0].endswith('.zst') else io.BufferedReader(counter)
            games = positions = duplicates = unknown_time_games = 0
            reasons = Counter()
            cap = source.get('max_positions', 0)
            with io.TextIOWrapper(reader, encoding='utf-8') as stream:
                while (game := chess.pgn.read_game(stream, Visitor=lambda: FilteredBuilder(source))) is not None:
                    games += 1; total_games += 1; progress()
                    reason = rejection(game, source)
                    if reason:
                        reasons[reason] += 1; rejected += 1
                        continue
                    if base_seconds(game.headers.get('TimeControl')) < 0:
                        unknown_time_games += 1
                    result = game.headers['Result']
                    board = game.board()
                    for ply, move in enumerate(game.mainline_moves()):
                        if ply >= 8 and ply % stride == 0:
                            keep = True
                            if db:
                                keep = db.execute('INSERT OR IGNORE INTO positions VALUES (?)', (position_key(board),)).rowcount == 1
                                if not keep:
                                    duplicates += 1
                            if keep:
                                row, legal = record(board)
                                row['policy'][legal.index(move)] = 1
                                win = result == ('1-0' if board.turn else '0-1')
                                row['value'][0 if win else 1 if result == '1/2-1/2' else 2] = 1
                                row['value_weight'] = 1
                                writer.add(split_for(board), row)
                                count += 1; positions += 1
                                if db and count % 4096 == 0:
                                    db.commit()
                        board.push(move)
                        if cap and positions >= cap:
                            break
                    if cap and positions >= cap:
                        break
            compressed += counter.count; counter.close(); counter = None
            summaries.append(dict(source, games=games, positions=positions, duplicates_removed=duplicates,
                                  rejection_counts=dict(reasons), accepted_unknown_time_games=unknown_time_games, reached_cap=bool(cap and positions >= cap)))
        provenance = dict(kind='human-pgn', profile=profile, sources=summaries, stride=stride,
            deduplicate_positions=deduplicate, games=total_games, rejected=rejected,
            compressed_bytes_read=compressed, split='position-hash-1-percent')
        if len(sources) == 1:
            provenance.update(source=sources[0]['url'], min_elo=sources[0]['min_elo'], max_positions=sources[0].get('max_positions',0))
        writer.close(provenance)
        (Path(out) / 'SOURCES.json').write_text(json.dumps(sources, indent=2))
        progress(force=True, phase='prepared')
    except BaseException as error:
        progress(force=True, phase='prepare_failed', error=str(error)); raise
    finally:
        if counter:
            counter.close()
        if db:
            db.commit(); db.close()


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--pgn', help='Local PGN/PGN.zst or HTTPS URL')
    source.add_argument('--profile', help='Versioned JSON source and quality-filter profile')
    p.add_argument('--out', required=True)
    p.add_argument('--min-elo', type=int, default=2000)
    p.add_argument('--min-base-seconds', type=int, default=0)
    p.add_argument('--max-positions', type=int, default=0)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--status-dir')
    a = p.parse_args()
    if a.stride < 1 or min(a.max_positions,a.min_elo,a.min_base_seconds) < 0:
        p.error('invalid sampling limits')
    profile = json.loads(Path(a.profile).read_text()) if a.profile else None
    sources = profile['sources'] if profile else [dict(url=a.pgn,min_elo=a.min_elo,min_base_seconds=a.min_base_seconds,max_positions=a.max_positions)]
    if not sources or any(min(s.get('min_elo',0),s.get('min_base_seconds',0),s.get('max_positions',0))<0 for s in sources):
        p.error('invalid source profile')
    prepare(sources, a.out, a.stride, a.status_dir, profile,
            deduplicate=profile.get('deduplicate_positions',False) if profile else False)


if __name__ == '__main__':
    main()
