import copy
from scripts.search_repair_run_cloud import gate_decision


def metrics():
 return dict(diagnostics=dict(correct=13),holdout=dict(correct=28),general=dict(ce=.5,mae=.2),balanced=dict(mae=.3,by_stratum={k:dict(mae=.3) for k in ['winning','neutral','losing']}),opponent_mean_value=-.18)


def test_only_all_gate_checks_enable_conditional_continuation():
 ref=metrics();candidate=copy.deepcopy(ref);candidate['balanced']['mae']=.25
 assert gate_decision(ref,candidate)['passed']
 for mutation in [lambda d:d['diagnostics'].update(correct=12),lambda d:d['holdout'].update(correct=26),lambda d:d['general'].update(ce=.52),lambda d:d['general'].update(mae=.23),lambda d:d['balanced']['by_stratum']['losing'].update(mae=.34),lambda d:d.update(opponent_mean_value=.01)]:
  candidate=copy.deepcopy(ref);mutation(candidate);decision=gate_decision(ref,candidate)
  assert not decision['passed'] and not all(decision['checks'].values())


def test_nan_metrics_never_pass_gate():
 ref=metrics();candidate=copy.deepcopy(ref);candidate['balanced']['mae']=float('nan')
 assert not gate_decision(ref,candidate)['passed']
