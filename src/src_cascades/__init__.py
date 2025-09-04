---
### **`src/src_cascades/__init__.py`**
---
```python
__version__ = "3.0.0"

from .distributions import BranchingDistribution, Poisson, FixedDegree, Geometric
from .dynamics import standard_intensity_dynamics, saturating_intensity_dynamics
from .simulator import SRCSimulator, NetworkSRCSimulator
from .solver import PGFSolver
from .results import SimulationResult, CascadeMetrics
from .analysis import run_parameter_sweep, analyze_traveling_wave, analyze_network_criticality
from .plotting import plot_size_distribution, plot_sweep_result, plot_temporal_evolution
from .runner import run_experiment_from_config
from .inference import Fitter