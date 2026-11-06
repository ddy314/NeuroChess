from pathlib import Path
import subprocess,tempfile
root=Path(__file__).resolve().parents[1]/'neurochess/native'
with tempfile.TemporaryDirectory(dir=root) as tmp:
    built=Path(tmp)/'libmcts_pipeline.so'
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-DNDEBUG','-fPIC','-shared','-Wall','-Wextra',str(root/'mcts_pipeline.cpp'),'-o',str(built)],check=True)
    built.replace(root/'libmcts_pipeline.so')
print(root/'libmcts_pipeline.so')
