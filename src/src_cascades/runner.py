import yaml
import pandas as pd
import numpy as np
import pickle
import json
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
from . import simulator, solver, distributions, dynamics, analysis, plotting, inference

DISTRIBUTION_FACTORY = {
    'Poisson': distributions.Poisson,
    'FixedDegree': distributions.FixedDegree,
    'Geometric': distributions.Geometric,
}
DYNAMICS_FACTORY = {
    'standard': dynamics.standard_intensity_dynamics,
    'saturating': dynamics.saturating_intensity_dynamics,
}

def create_distribution(config: dict):
    name = config['name']
    params = config.get('params', {})
    return DISTRIBUTION_FACTORY[name](**params)

def create_dynamics(config: dict):
    name = config['name']
    params = config.get('params', {})
    return DYNAMICS_FACTORY[name](**params)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def run_experiment_from_config(config_path: str):
    """Loads a YAML config, runs the specified experiment, and saves the results."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_config = config['experiment']
    output_config = config['output']
    
    output_dir = Path(output_config['directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dist = create_distribution(experiment_config['distribution'])
    dynamic_func = create_dynamics(experiment_config['dynamics'])
    
    task = experiment_config['task']
    results = None
    if task in ['single_run', 'parameter_sweep']:
        if task == 'single_run':
            sim = simulator.SRCSimulator(
                p=experiment_config['params']['p'], distribution=dist,
                dynamics=dynamic_func, backend=experiment_config.get('backend', 'python')
            )
            results = sim.run_simulations(**experiment_config['run_settings'])
        
        elif task == 'parameter_sweep':
            def target_func(p, distribution, num_simulations, backend, num_cores):
                sim = simulator.SRCSimulator(p=p, distribution=distribution, backend=backend)
                return sim.run_simulations(num_simulations=num_simulations, num_cores=num_cores)
            
            results = analysis.run_parameter_sweep(
                target_func=target_func, sweep_params=experiment_config['sweep_params'],
                fixed_params={ 'distribution': dist, **experiment_config['run_settings']}
            )
    
    elif task in ['fit_mle', 'fit_mcmc']:
        data_path = experiment_config['data_path']
        with open(data_path, 'r') as f:
            data = [int(line.strip()) for line in f if line.strip()]
        
        fitter = inference.Fitter(data, experiment_config)
        
        if task == 'fit_mle':
            results = fitter.fit_mle(**experiment_config['mle_settings'])
            results['optimizer_result'] = str(results['optimizer_result']) # Make serializable
            with open(output_dir / output_config['results_file'], 'w') as f:
                json.dump(results, f, indent=2, cls=NpEncoder)
        
        elif task == 'fit_mcmc':
            results = fitter.fit_mcmc(**experiment_config['mcmc_settings'])
            results.to_netcdf(output_dir / output_config['results_file'])
    
    else:
        raise ValueError(f"Unknown task in config: {task}")
    
    print(f"Results saved to {output_dir / output_config['results_file']}")

    if output_config.get('generate_plots', False):
        if task == 'single_run':
            fig, ax = plt.subplots(figsize=(10, 7))
            plotting.plot_size_distribution(results, ax=ax)
            ax.set_title(f"Cascade size distribution (p={results.configuration['p']})")
            fig.savefig(output_dir / "size_distribution.png")
            
        elif task == 'parameter_sweep':
            param_name = list(experiment_config['sweep_params'].keys())[0]
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            plotting.plot_sweep_result(results, param_name, 'mean_size', ax=axes[0])
            plotting.plot_sweep_result(results, param_name, 'mean_depth', ax=axes[1])
            fig.suptitle(f"Parameter sweep over '{param_name}'")
            fig.savefig(output_dir / "parameter_sweep.png")
            
        elif task == 'fit_mcmc':
            az.plot_trace(results)
            plt.tight_layout()
            plt.savefig(output_dir / "mcmc_trace.png")
            
            az.plot_posterior(results)
            plt.tight_layout()
            plt.savefig(output_dir / "mcmc_posterior.png")
        
        print(f"Plots saved in {output_dir}")