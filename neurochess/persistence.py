"""Copy local training artifacts to an existing mounted persistent directory."""
import os
from pathlib import Path
import shutil


class RunMirror:
    def __init__(self, source, destination, mount):
        self.source = Path(source)
        self.destination = Path(destination)
        self.mount = Path(mount)
        self.copied = {}

    def sync(self):
        # Never recreate a disappeared mount or silently save on ephemeral storage.
        if not os.path.ismount(self.mount) or not self.destination.is_dir():
            raise OSError('Persistent mount or original run directory is unavailable')
        paths = sorted(p for p in self.source.rglob('*') if p.is_file()
                       and p.suffix != '.tmp')
        # Publish completion only after the checkpoint and other artifacts.
        paths.sort(key=lambda p: p.name == 'status.json')
        for path in paths:
            relative = path.relative_to(self.source)
            target = self.destination / relative
            with path.open('rb') as src:
                stat = os.fstat(src.fileno())
                signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if self.copied.get(relative) == signature:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.mirror.tmp')
                with temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                if temporary.stat().st_size != stat.st_size:
                    # Appending logs may grow during copying. Retry a fresh snapshot.
                    temporary.unlink()
                    continue
                temporary.replace(target)
                self.copied[relative] = signature
