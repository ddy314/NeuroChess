"""Use an existing CUDA interpreter without changing its installed packages."""
import importlib.util
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
if importlib.util.find_spec('chess') is None:
    for site in (root / '.venv/lib').glob('python*/site-packages'):
        if (site / 'chess/__init__.py').is_file():
            sys.path.append(str(site))
            break
runpy.run_module('neurochess.train', run_name='__main__')
