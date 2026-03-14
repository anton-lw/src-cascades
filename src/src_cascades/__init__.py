"""Public package exports for src_cascades."""

__version__ = "3.0.0"

from .analysis import (
    analyze_network_criticality,
    analyze_traveling_wave,
    run_parameter_sweep,
)
from .distributions import BranchingDistribution, FixedDegree, Geometric, Poisson
from .dynamics import standard_intensity_dynamics, saturating_intensity_dynamics
from .inference import Fitter
from .plotting import plot_size_distribution, plot_sweep_result, plot_temporal_evolution
from .results import CascadeMetrics, SimulationResult
from .runner import run_experiment_from_config
from .simulator import NetworkSRCSimulator, SRCSimulator
from .solver import PGFSolver

__all__ = [
    "__version__",
    "BranchingDistribution",
    "Poisson",
    "FixedDegree",
    "Geometric",
    "standard_intensity_dynamics",
    "saturating_intensity_dynamics",
    "SRCSimulator",
    "NetworkSRCSimulator",
    "PGFSolver",
    "SimulationResult",
    "CascadeMetrics",
    "run_parameter_sweep",
    "analyze_traveling_wave",
    "analyze_network_criticality",
    "plot_size_distribution",
    "plot_sweep_result",
    "plot_temporal_evolution",
    "run_experiment_from_config",
    "Fitter",
]
