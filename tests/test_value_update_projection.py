import torch
from scripts.local_value_sustained import project_update


def test_project_actual_update_preserves_safe_displacement():
    g = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    d = torch.tensor([-2., -3., 4.])
    projected, _ = project_update(d, g)
    assert torch.equal(projected, d)


def test_project_conflicting_constraints_keeps_useful_orthogonal_learning():
    g = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    d = torch.tensor([2., 3., 4.])
    projected, info = project_update(d, g)
    assert torch.all(g @ projected <= 1e-7)
    assert torch.allclose(projected, torch.tensor([0., 0., 4.]), atol=1e-6)
    assert info['max_violation'] <= 1e-7


def test_projection_handles_correlated_gradients_and_zero_gradients():
    torch.manual_seed(8)
    g = torch.randn(10, 50); g[0] = 0; g[2] = g[1]*2
    d = torch.randn(50)
    projected, _ = project_update(d, g)
    assert torch.max(g @ projected).item() < 1e-5
    assert projected.norm() > 0
