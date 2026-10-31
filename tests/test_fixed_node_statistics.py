import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('fixed_stats',Path('scripts/summarize_fixed_node_arena.py'))
stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)


def test_exact_pair_bootstrap_and_elo_sign():
    assert stats.exact_pair_bootstrap([.25,.75])==[.25,.75]
    assert stats.exact_pair_bootstrap([.5]*16)==[.5,.5]
    assert stats.rating(.75)>0 and stats.rating(.25)<0 and stats.rating(.5)==0


def test_joint_ladder_recovers_known_relative_ratings():
    ratings=np.array([60.,-200.,0.,100.]);scale=400/np.log(10)
    probability=1/(1+np.exp((ratings[None,:]-ratings[:,None])/scale))
    wins=32*probability;np.fill_diagonal(wins,0)
    np.testing.assert_allclose(stats.fit_ladder(wins),ratings,atol=1e-5)


def test_separated_opponent_is_unbounded_instead_of_fake_finite_rating():
    wins=np.ones((4,4));np.fill_diagonal(wins,0);wins[1,:]=0
    estimate=stats.fit_ladder(wins)
    assert np.isneginf(estimate[1]) and estimate[2]==0
