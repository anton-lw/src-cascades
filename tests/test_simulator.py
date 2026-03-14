import numpy as np
import pytest
import networkx as nx

from src_cascades.distributions import FixedDegree, Poisson
from src_cascades.dynamics import saturating_intensity_dynamics
from src_cascades.simulator import NetworkSRCSimulator, SRCSimulator

def test_run_cascade_honors_termination_and_history():
    simulator = SRCSimulator(p=0.5, distribution=FixedDegree(0))
    metrics = simulator.run_cascade(initial_intensity=3, record_history=True)

    assert metrics.size == 1
    assert metrics.depth == 1
    assert metrics.max_width == 1
    assert metrics.total_intensity_effort == 3
    assert metrics.temporal_history == {0: {"nodes": 1, "intensities": [3]}}

def test_run_cascade_stops_immediately_for_non_positive_initial_intensity():
    simulator = SRCSimulator(p=0.5, distribution=FixedDegree(2))
    metrics = simulator.run_cascade(initial_intensity=0)

    assert metrics.size == 0
    assert metrics.depth == 0
    assert metrics.max_width == 0

def test_run_cascade_respects_max_steps():
    simulator = SRCSimulator(p=1.0, distribution=FixedDegree(2))
    metrics = simulator.run_cascade(initial_intensity=1, max_steps=5)

    assert metrics.size == 5

def test_parallel_simulations_work_for_pickle_safe_dynamics():
    simulator = SRCSimulator(
        p=0.1,
        distribution=Poisson(2.0),
        dynamics=saturating_intensity_dynamics(5),
    )
    result = simulator.run_simulations(num_simulations=8, num_cores=2, initial_intensity=1)

    assert len(result.run_metrics) == 8
    assert result.mean_size > 0

def test_parallel_simulations_fall_back_for_non_picklable_callbacks():
    simulator = SRCSimulator(
        p=0.1,
        distribution=Poisson(2.0),
        termination_condition=lambda intensity: intensity <= 0,
    )

    with pytest.warns(RuntimeWarning, match="Falling back to serial execution"):
        result = simulator.run_simulations(num_simulations=6, num_cores=2, initial_intensity=1)

    assert len(result.run_metrics) == 6

def test_network_simulator_visits_expected_nodes():
    graph = nx.path_graph(5)
    simulator = NetworkSRCSimulator(graph=graph, p=1.0)
    metrics = simulator.run_cascade(start_node=0, initial_intensity=1)

    assert metrics.size == 5
    assert metrics.depth == 4
    assert metrics.max_width == 1
    assert metrics.total_intensity_effort == 15.0

def test_parallel_runs_are_not_all_identical():
    simulator = SRCSimulator(p=0.1, distribution=Poisson(2.0))
    result = simulator.run_simulations(num_simulations=10, num_cores=2, initial_intensity=1)

    assert len(np.unique(result.cascade_sizes)) > 1
