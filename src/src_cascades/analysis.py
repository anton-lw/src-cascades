import pandas as pd
import numpy as np
import sympy as sp
import itertools
import networkx as nx
from typing import Any, List, Dict, Callable

def analyze_traveling_wave(p: float, ell: float) -> Dict[str, float]:
    """
    Solves the transcendental equations for the traveling wave properties.
    Based on Eqs. (7) and (9) of the paper.
    """
    pc_theory = 0.5 - np.sqrt(max(0, ell**2 - 1)) / (2 * ell) if ell >= 1 else np.inf
    if p > pc_theory:
        x = sp.symbols('x')
        try:
            mu_eqn = sp.log(ell * (1 - p) + ell * p * sp.exp(x)) - p * x * sp.exp(x) / (1 - p + p * sp.exp(x))
            mu = float(sp.nsolve(mu_eqn, 3.0, solver='bisect', tol=1e-6))
            v_max = p * np.exp(mu) / (1 - p + p * np.exp(mu))
            n_c = (3.0 / mu) * (1 / (2.0 * v_max - 1)) if abs(2 * v_max - 1) > 1e-9 else np.inf
            return {"mu": mu, "v_max": v_max, "n_c": n_c}
        except (ValueError, TypeError):
             return {"mu": np.nan, "v_max": np.nan, "n_c": np.nan}
    else:
        return {"mu": np.nan, "v_max": np.nan, "n_c": np.nan}

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