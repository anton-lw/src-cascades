import numpy as np
import networkx as nx
from collections import deque
from multiprocessing import Pool, cpu_count
from typing import Callable, Optional, Any, List
from . import distributions, dynamics
from .results import SimulationResult, CascadeMetrics

try:
    from . import simulator_backend_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ backend not found. Falling back to pure Python.")

def _run_single_cascade_wrapper(args):
    """Helper for multiprocessing to call instance methods."""
    simulator_instance, kwargs = args
    return simulator_instance.run_cascade(**kwargs)

class SRCSimulator:
    """Simulates Self-Reinforcing Cascades on an abstract branching process."""
    def __init__(self,
                 p: float,
                 distribution: distributions.BranchingDistribution,
                 dynamics: dynamics.IntensityDynamics = dynamics.standard_intensity_dynamics,
                 termination_condition: Callable[[int], bool] = lambda k: k <= 0,
                 backend: str = 'python'):
        self.p = p
        self.distribution = distribution
        self.dynamics = dynamics
        self.termination_condition = termination_condition
        self.rng = np.random.default_rng()
        
        if backend == 'cpp':
            if not CPP_AVAILABLE:
                raise ImportError("C++ backend selected, but compiled module is not available.")
            if not isinstance(self.distribution, distributions.Poisson):
                raise ValueError("C++ backend currently only supports Poisson distribution.")
        self.backend = backend

    def _run_cascade_python(self, initial_intensity: int, record_history: bool, max_steps: int) -> CascadeMetrics:
        """Pure Python implementation of the cascade simulation."""
        if self.termination_condition(initial_intensity):
            return CascadeMetrics(size=0, depth=0, max_width=0, total_intensity_effort=0)

        current_gen_q = deque([initial_intensity])
        next_gen_q = deque()
        total_size, depth, max_width, total_intensity_effort = 0, 0, 0, 0.0
        history = {} if record_history else None

        while current_gen_q and total_size < max_steps:
            gen_width = len(current_gen_q)
            max_width = max(max_width, gen_width)
            if record_history:
                history[depth] = {'nodes': gen_width, 'intensities': list(current_gen_q)}

            for _ in range(gen_width):
                current_intensity = current_gen_q.popleft()
                total_size += 1
                total_intensity_effort += current_intensity
                num_children = self.distribution.sample()
                for _ in range(num_children):
                    is_receptive = self.rng.random() < self.p
                    new_intensity = self.dynamics(current_intensity, is_receptive)
                    if not self.termination_condition(new_intensity):
                        next_gen_q.append(new_intensity)
            
            current_gen_q = next_gen_q
            next_gen_q = deque()
            depth += 1
        
        return CascadeMetrics(
            size=total_size, depth=depth, max_width=max_width,
            total_intensity_effort=total_intensity_effort,
            temporal_history=history
        )

    def _run_cascade_cpp(self, initial_intensity: int, record_history: bool, max_steps: int) -> CascadeMetrics:
        """Wrapper to call the compiled C++ function."""
        result_dict = simulator_backend_cpp.run_cascade_cpp(
            self.p, self.distribution.mean(), initial_intensity, 
            record_history, max_steps, self.dynamics, self.termination_condition
        )
        return CascadeMetrics(**result_dict)

    def run_cascade(self, initial_intensity: int = 1, record_history: bool = False, max_steps: int = 100000) -> CascadeMetrics:
        """Dispatcher method that selects the backend to run a single cascade."""
        if self.backend == 'cpp':
            return self._run_cascade_cpp(initial_intensity, record_history, max_steps)
        else:
            return self._run_cascade_python(initial_intensity, record_history, max_steps)

    def run_simulations(self, num_simulations: int, num_cores: Optional[int] = None, **kwargs) -> SimulationResult:
        """Runs multiple simulations in parallel, returning a rich result object."""
        if num_cores is None:
            num_cores = max(1, cpu_count() - 1)

        config = {
            "p": self.p, "distribution": self.distribution.__class__.__name__,
            "ell": self.distribution.mean() if isinstance(self.distribution, distributions.Poisson) else None,
            "dynamics": self.dynamics.__name__, "num_simulations": num_simulations,
            "backend": self.backend, **kwargs
        }

        if num_cores > 1 and num_simulations > 1:
            with Pool(num_cores) as pool:
                args_list = [(self, kwargs) for _ in range(num_simulations)]
                metrics_list = pool.map(_run_single_cascade_wrapper, args_list)
        else:
            metrics_list = [self.run_cascade(**kwargs) for _ in range(num_simulations)]
        
        return SimulationResult(configuration=config, run_metrics=metrics_list)

class NetworkSRCSimulator:
    """Simulates Self-Reinforcing Cascades on a user-provided NetworkX graph."""
    def __init__(self,
                 graph: nx.Graph,
                 p: float,
                 dynamics: dynamics.IntensityDynamics = dynamics.standard_intensity_dynamics,
                 termination_condition: Callable[[int], bool] = lambda k: k <= 0):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.p = p
        self.dynamics = dynamics
        self.termination_condition = termination_condition
        self.rng = np.random.default_rng()

    def run_cascade(self, start_node: Any, initial_intensity: int = 1) -> CascadeMetrics:
        """Runs a single cascade on the graph, returning detailed metrics."""
        if self.termination_condition(initial_intensity):
            return CascadeMetrics(size=0, depth=0, max_width=0, total_intensity_effort=0)
        if start_node not in self.graph:
            raise ValueError("Start node not in graph.")

        queue = deque([(start_node, initial_intensity, 0)])
        visited = {start_node: 0}
        total_intensity_effort = float(initial_intensity)
        nodes_at_depth = {0: 1}

        while queue:
            current_node, current_intensity, current_depth = queue.popleft()
            
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    is_receptive = self.rng.random() < self.p
                    new_intensity = self.dynamics(current_intensity, is_receptive)
                    
                    if not self.termination_condition(new_intensity):
                        new_depth = current_depth + 1
                        visited[neighbor] = new_depth
                        nodes_at_depth[new_depth] = nodes_at_depth.get(new_depth, 0) + 1
                        total_intensity_effort += new_intensity
                        queue.append((neighbor, new_intensity, new_depth))

        max_depth = max(nodes_at_depth.keys()) if nodes_at_depth else 0
        max_width = max(nodes_at_depth.values()) if nodes_at_depth else 0

        return CascadeMetrics(
            size=len(visited),
            depth=max_depth,
            max_width=max_width,
            total_intensity_effort=total_intensity_effort,
            temporal_history=None
        )

    def run_simulations(self, num_simulations: int, num_cores: Optional[int] = None, start_node_strategy: str = 'random', **kwargs) -> SimulationResult:
        """Runs multiple simulations on the graph."""
        if start_node_strategy != 'random':
            raise NotImplementedError("Only 'random' start node strategy is implemented.")
        
        metrics_list: List[CascadeMetrics] = []
        for _ in range(num_simulations):
            start_node = self.rng.choice(self.nodes)
            kwargs['start_node'] = start_node
            metrics_list.append(self.run_cascade(**kwargs))

        config = {"p": self.p, "graph_type": self.graph.__class__.__name__, "num_simulations": num_simulations, **kwargs}
        return SimulationResult(configuration=config, run_metrics=metrics_list)