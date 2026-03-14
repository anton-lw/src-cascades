import math

import numpy as np

from src_cascades.distributions import Poisson
from src_cascades.simulator import SRCSimulator
from src_cascades.solver import PGFSolver

def test_expected_size_matches_subcritical_monte_carlo():
    solver = PGFSolver(p=0.05, distribution=Poisson(2.0))
    simulator = SRCSimulator(p=0.05, distribution=Poisson(2.0))

    expected_size = solver.get_expected_size(initial_intensity=1)
    simulations = simulator.run_simulations(num_simulations=5000, num_cores=1, initial_intensity=1)

    assert math.isclose(expected_size, 1.5505102572168217, rel_tol=1e-12)
    assert abs(simulations.mean_size - expected_size) < 0.2

def test_supercritical_probability_matches_distribution_mass_and_simulation():
    solver = PGFSolver(p=0.1, distribution=Poisson(2.0))
    simulator = SRCSimulator(p=0.1, distribution=Poisson(2.0))

    p_infinite = solver.get_supercritical_prob(k_max=80, max_iter=2000)
    size_distribution = solver.get_size_distribution(max_size=256, k_max=80, max_iter=2000)
    simulations = simulator.run_simulations(
        num_simulations=2000,
        num_cores=1,
        initial_intensity=1,
        max_steps=5000,
    )
    simulated_survival = np.mean(simulations.cascade_sizes >= 5000)

    assert 0 < p_infinite < 1
    assert math.isclose(size_distribution.sum(), 1 - p_infinite, rel_tol=1e-9, abs_tol=1e-9)
    assert abs(simulated_survival - p_infinite) < 0.02

def test_expected_size_diverges_in_supercritical_regime():
    solver = PGFSolver(p=0.1, distribution=Poisson(2.0))

    assert math.isinf(solver.get_expected_size(initial_intensity=1))
