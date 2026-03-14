import pickle

import numpy as np

from src_cascades.distributions import FixedDegree, Geometric, Poisson
from src_cascades.dynamics import standard_intensity_dynamics, saturating_intensity_dynamics

def test_distribution_means_and_pgfs():
    poisson = Poisson(ell=2.5)
    fixed = FixedDegree(k=3)
    geometric = Geometric(p=0.25)

    x = np.array([0.2, 0.5, 0.9])

    assert poisson.mean() == 2.5
    assert np.allclose(poisson.pgf(x), np.exp(2.5 * (x - 1)))

    assert fixed.mean() == 3.0
    assert np.allclose(fixed.pgf(x), x**3)

    assert geometric.mean() == 3.0
    assert np.allclose(geometric.pgf(x), 0.25 / (1 - 0.75 * x))

def test_standard_and_saturating_dynamics():
    saturating = saturating_intensity_dynamics(2)

    assert standard_intensity_dynamics(3, True) == 4
    assert standard_intensity_dynamics(3, False) == 2

    assert saturating(1, True) == 2
    assert saturating(2, True) == 2
    assert saturating(2, False) == 1

def test_saturating_dynamics_is_pickle_safe():
    dynamics = saturating_intensity_dynamics(4)

    pickle.dumps(dynamics)
