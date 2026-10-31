"""Compile the local CPU evaluator and best-first minimax search."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def build(debug=False):
    target = ROOT/'neurochess/native/libbestfirst.so'
    flags = ['-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all'] if debug else [
        '-O3', '-march=native', '-DNDEBUG']
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        built = Path(tmp)/target.name
        subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra',
                        *flags, str(target.with_name('best_first.cpp')), '-o', str(built)], check=True)
        built.replace(target)
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--debug', action='store_true')
    print(build(p.parse_args().debug))
