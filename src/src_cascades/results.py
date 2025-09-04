from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class CascadeMetrics:
    """Holds the detailed metrics for a single cascade run."""
    size: int
    depth: int
    max_width: int
    total_intensity_effort: float
    temporal_history: Optional[Dict[int, Dict[str, Any]]] = None # {gen: {'nodes': count, 'intensities': [..]}}

@dataclass
class SimulationResult:
    """A data class to hold aggregated results and configuration of a simulation run."""
    configuration: Dict[str, Any]
    run_metrics: List[CascadeMetrics]
    
    cascade_sizes: np.ndarray = field(init=False)
    depths: np.ndarray = field(init=False)
    max_widths: np.ndarray = field(init=False)
    
    mean_size: float = field(init=False)
    mean_depth: float = field(init=False)
    mean_max_width: float = field(init=False)

    def __post_init__(self):
        if not self.run_metrics:
            self.cascade_sizes = np.array([])
            self.depths = np.array([])
            self.max_widths = np.array([])
            self.mean_size = 0.0
            self.mean_depth = 0.0
            self.mean_max_width = 0.0
        else:
            self.cascade_sizes = np.array([m.size for m in self.run_metrics])
            self.depths = np.array([m.depth for m in self.run_metrics])
            self.max_widths = np.array([m.max_width for m in self.run_metrics])
            
            self.mean_size = np.mean(self.cascade_sizes)
            self.mean_depth = np.mean(self.depths)
            self.mean_max_width = np.mean(self.max_widths)