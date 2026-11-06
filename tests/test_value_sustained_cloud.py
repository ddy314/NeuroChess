import copy
import torch
from scripts.value_sustained_cloud import numeric_checks, state_digest


def test_original_numeric_anchor_cannot_drift_with_later_checkpoints():
    reference = dict(general=dict(ce=.5,mae=.2),balanced=dict(mae=.4,
        by_stratum={k:dict(mae=.3) for k in ['winning','neutral','losing']}))
    candidate = copy.deepcopy(reference)
    assert all(numeric_checks(reference,candidate).values())
    candidate['general']['ce'] = .516
    assert not numeric_checks(reference,candidate)['general_ce']
    candidate = copy.deepcopy(reference); candidate['balanced']['by_stratum']['neutral']['mae'] = .331
    assert not numeric_checks(reference,candidate)['per_value_stratum']
    candidate = copy.deepcopy(reference); candidate['general']['ce'] = float('nan')
    assert not numeric_checks(reference,candidate)['general_ce']


def test_transport_digest_checks_tensor_content_and_dtype_independent_of_file_metadata():
    state = {'b':torch.tensor([1.,2.]),'a':torch.tensor([[3.]])}
    same = {'a':state['a'].clone(),'b':state['b'].clone()}
    assert state_digest(state)==state_digest(same)
    same['b'][0] += 1
    assert state_digest(state)!=state_digest(same)
    same = {k:v.double() for k,v in state.items()}
    assert state_digest(state)!=state_digest(same)
