import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('floor',Path(__file__).parents[1]/'scripts/check_search_floor.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def report(error=0,seconds=.18):
    def row(name,cp,t):return dict(method=name,cp_regret=cp,seconds=t,teacher=dict(cp=0,mate=None))
    return dict(positions=[dict(index=i,methods=[row('baseline',0,1.98),row('fast',error,seconds)]) for i in range(16)])


def test_floor_rejects_fast_but_weaker_and_does_not_promote_without_arena():
    bad=f.evaluate(report(150),'fast');assert bad['speed_ok'] and not bad['position_ok'] and not bad['promote']
    same=f.evaluate(report(),'fast');assert same['speed_ok'] and same['position_ok'] and not same['arena_ok'] and not same['promote']
    slow=f.evaluate(report(seconds=.3),'fast');assert not slow['speed_ok']
