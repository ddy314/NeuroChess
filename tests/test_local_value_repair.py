import copy
from scripts.local_value_repair import eligible


def reference():
 return {'general':{'ce':.373,'by_stratum':{'losing':{'mae':.27}}},'balanced':{'mae':.69,'by_stratum':{k:{'mae':.5} for k in ['winning','neutral','losing']}}}


def test_average_repair_improvement_cannot_hide_general_losing_degradation():
 r=reference();c=copy.deepcopy(r);c['balanced']['mae']=.60
 assert eligible(r,c)
 c['general']['by_stratum']['losing']['mae']=.28
 assert not eligible(r,c)
 c=copy.deepcopy(r);c['general']['ce']=.38
 assert not eligible(r,c)
 c=copy.deepcopy(r);c['balanced']['by_stratum']['neutral']['mae']=.52
 assert not eligible(r,c)
