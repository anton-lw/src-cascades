import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from .results import SimulationResult, CascadeMetrics

def plot_size_distribution(
    result: SimulationResult, ax=None, plot_sim=True, plot_pgf=None,
    label_sim="Simulation", label_pgf="PGF Solver", **kwargs
):
    """Plots the cascade size distribution from simulation and/or PGF results."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    if plot_sim and result.cascade_sizes is not None and len(result.cascade_sizes) > 0:
        counts = Counter(result.cascade_sizes)
        total = len(result.cascade_sizes)
        sizes, probs = zip(*sorted(counts.items()))
        probs = np.array(probs) / total
        ax.plot(sizes, probs, 'o', alpha=0.7, label=label_sim, **kwargs)

    if plot_pgf is not None:
        ax.plot(np.arange(len(plot_pgf)), plot_pgf, '-', linewidth=2.5, label=label_pgf, **kwargs)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Cascade Size (s)")
    ax.set_ylabel("Probability P(s)")
    ax.legend()
    ax.grid(True, which="both", ls="--", c='0.7')
    return ax

def plot_sweep_result(df: pd.DataFrame, param_name: str, metric_name: str, ax=None, **kwargs):
    """Plots a single metric from a parameter sweep DataFrame."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(df[param_name], df[metric_name], '-o', **kwargs)
    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} vs. {param_name.replace('_', ' ').title()}")
    ax.grid(True, ls="--", c='0.7')
    return ax

def plot_temporal_evolution(metrics: CascadeMetrics, ax=None, **kwargs):
    """Plots the temporal evolution (width vs. depth) of a single cascade."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if metrics.temporal_history:
        history = metrics.temporal_history
        generations = list(history.keys())
        nodes_per_gen = [h['nodes'] for h in history.values()]
        ax.plot(generations, nodes_per_gen, '-o', **kwargs)
        ax.set_xlabel("Generation (depth)")
        ax.set_ylabel("Number of active nodes (width)")
        ax.set_title("Temporal evolution of a single cascade")
        ax.grid(True, ls="--", c='0.7')
    else:
        ax.text(0.5, 0.5, "No temporal history recorded.", ha='center', va='center', transform=ax.transAxes)
        
    return ax