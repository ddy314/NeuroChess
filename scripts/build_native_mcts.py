"""Build with the existing system compiler; no package/environment installation."""
import argparse
import pathlib
import tempfile
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

def build(debug=False):
    target = ROOT / 'neurochess/native/libmcts.so'
    flags = ['-O3', '-march=native', '-DNDEBUG'] if not debug else ['-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all']
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        built = pathlib.Path(tmp) / target.name
        subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', *flags,
                        str(ROOT / 'neurochess/native/mcts.cpp'), '-o', str(built)], check=True)
        built.replace(target)
    return target

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--debug', action='store_true')
    print(build(p.parse_args().debug))
