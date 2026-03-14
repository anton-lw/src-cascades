import math
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from typing import Any, List, Dict, Callable
from scipy.optimize import root_scalar

def critical_point(ell: float) -> float:
    """Critical point from Eq. (4) of the paper."""
    if ell < 1:
        return np.inf
    return 0.5 * (1 - np.sqrt(1 - 1 / (ell**2)))

def analyze_traveling_wave(p: float, ell: float) -> Dict[str, float]:
    """
    Solves the traveling-wave quantities from Eqs. (6) and (10).
    """
    pc_theory = critical_point(ell)
    if ell <= 0 or not (0 < p < 1) or p <= pc_theory:
        return {"mu": np.nan, "v_max": np.nan, "n_c": np.nan}

    def velocity_selection(mu: float) -> float:
        exp_mu = math.exp(mu)
        denom = 1 - p + p * exp_mu
        return math.log(ell * denom) - mu * (p * exp_mu / denom)

    bracket = None
    search_grid = np.concatenate([
        np.linspace(1e-8, 1e-2, 250),
        np.logspace(-2, 2, 400),
    ])
    prev_mu = float(search_grid[0])
    prev_val = velocity_selection(prev_mu)
    for mu in search_grid[1:]:
        current_mu = float(mu)
        current_val = velocity_selection(current_mu)
        if prev_val == 0:
            bracket = (prev_mu, prev_mu)
            break
        if current_val == 0 or prev_val * current_val < 0:
            bracket = (prev_mu, current_mu)
            break
        prev_mu = current_mu
        prev_val = current_val

    if bracket is None:
        return {"mu": np.nan, "v_max": np.nan, "n_c": np.nan}

    try:
        if bracket[0] == bracket[1]:
            mu = bracket[0]
        else:
            solution = root_scalar(velocity_selection, bracket=bracket, method="brentq")
            if not solution.converged:
                return {"mu": np.nan, "v_max": np.nan, "n_c": np.nan}
            mu = float(solution.root)
    except ValueError:
        return {"mu": np.nan, "v_max": np.nan, "n_c": np.nan}

    exp_mu = math.exp(mu)
    denom = 1 - p + p * exp_mu
    v_max = math.log(ell * denom) / mu
    if (2 * v_max) <= 1:
        n_c = np.inf
    else:
        n_c = 3.0 / (mu * (2.0 * v_max - 1.0))
    return {"mu": mu, "v_max": v_max, "n_c": n_c}

def analyze_network_criticality(graph: nx.Graph, p: float) -> dict:
    """
    Estimates the critical point using a mean-field approach on a network.
    """
    if not nx.is_connected(graph):
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
    else:
        subgraph = graph

    degrees = [d for n, d in subgraph.degree()]
    if not degrees:
        return {"effective_ell": 0, "p_c_network": np.inf, "is_supercritical": False}

    mean_degree = np.mean(degrees)
    mean_sq_degree = np.mean(np.square(degrees))
    
    effective_ell = (mean_sq_degree / mean_degree) - 1 if mean_degree > 0 else 0
    
    if effective_ell < 1:
        p_c = np.inf
    else:
        p_c = 0.5 - np.sqrt(effective_ell**2 - 1) / (2 * effective_ell)
        
    is_supercritical = p > p_c

    return {
        "effective_ell": effective_ell,
        "p_c_network": p_c,
        "is_supercritical": is_supercritical
    }

def run_parameter_sweep(
    target_func: Callable,
    sweep_params: Dict[str, List[Any]],
    fixed_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Runs a target function over a grid of parameters.
    """
    param_names = list(sweep_params.keys())
    param_value_lists = list(sweep_params.values())
    results = []

    for param_combination in itertools.product(*param_value_lists):
        current_params = dict(zip(param_names, param_combination))
        run_config = {**fixed_params, **current_params}
        
        print(f"Running for: {current_params}")
        result = target_func(**run_config)
        
        if isinstance(result, dict):
             output_data = result
        elif hasattr(result, '__dict__'):
            output_data = {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
        else:
            output_data = {'result': result}

        output_data.pop('run_metrics', None)
        output_data.pop('configuration', None)
        output_data.pop('cascade_sizes', None)
        output_data.pop('depths', None)
        output_data.pop('max_widths', None)

        results.append({**current_params, **output_data})

    return pd.DataFrame(results)
