import unittest
import torch
from scripts.lichess_manual_weight_probe import interpolate_state

class WeightEditTests(unittest.TestCase):
    def test_endpoints_and_original_tensors_untouched(self):
        a={'weight':torch.tensor([1.,-2.])};b={'weight':torch.tensor([3.,2.])}
        torch.testing.assert_close(interpolate_state(a,b,0)['weight'],a['weight'])
        torch.testing.assert_close(interpolate_state(a,b,1)['weight'],b['weight'])
        torch.testing.assert_close(interpolate_state(a,b,.25)['weight'],torch.tensor([1.5,-1.]))
        torch.testing.assert_close(a['weight'],torch.tensor([1.,-2.]))
        torch.testing.assert_close(b['weight'],torch.tensor([3.,2.]))
    def test_mismatches_nonfinite_and_extrapolation_rejected(self):
        for a,b,alpha in [({'x':torch.ones(2)},{'x':torch.ones(3)},.5),
            ({'x':torch.ones(2)},{'x':torch.tensor([float('nan'),1.])},.5),
            ({'x':torch.ones(2)},{'x':torch.ones(2)},1.1)]:
            with self.assertRaises(ValueError):interpolate_state(a,b,alpha)

if __name__=='__main__':unittest.main()
