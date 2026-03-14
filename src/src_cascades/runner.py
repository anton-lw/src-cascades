import yaml
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict
from . import simulator, distributions, dynamics, analysis, plotting

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
    if name not in DISTRIBUTION_FACTORY:
        raise ValueError(f"Unknown distribution: {name}")
    return DISTRIBUTION_FACTORY[name](**params)

def create_dynamics(config: dict):
    name = config['name']
    params = config.get('params', {})
    if name not in DYNAMICS_FACTORY:
        raise ValueError(f"Unknown dynamics: {name}")
    factory = DYNAMICS_FACTORY[name]
    return factory if name == 'standard' else factory(**params)

def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path

def _load_inference_module():
    from . import inference

    return inference

def _load_arviz():
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "Plotting MCMC results requires the optional inference dependencies. "
            "Install the 'inference' extra to enable this path."
        ) from exc
    return az

def _save_results(results, output_path: Path):
    suffix = output_path.suffix.lower()

    if isinstance(results, pd.DataFrame):
        if suffix == ".csv":
            results.to_csv(output_path, index=False)
            return
        if suffix == ".json":
            output_path.write_text(
                json.dumps(json.loads(results.to_json(orient="records")), indent=2),
                encoding="utf-8",
            )
            return

    if hasattr(results, "run_metrics") and hasattr(results, "configuration"):
        payload = {
            "configuration": results.configuration,
            "summary": {
                "mean_size": float(results.mean_size),
                "mean_depth": float(results.mean_depth),
                "mean_max_width": float(results.mean_max_width),
                "num_runs": len(results.run_metrics),
            },
            "run_metrics": [asdict(metric) for metric in results.run_metrics],
        }
        if suffix == ".json":
            output_path.write_text(json.dumps(payload, indent=2, cls=NpEncoder), encoding="utf-8")
            return
        if suffix == ".csv":
            pd.DataFrame(payload["run_metrics"]).to_csv(output_path, index=False)
            return

    if isinstance(results, dict):
        output_path.write_text(json.dumps(results, indent=2, cls=NpEncoder), encoding="utf-8")
        return

    raise ValueError(f"Unsupported output format for {output_path.name}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def run_experiment_from_config(config_path: str):
    """Loads a YAML config, runs the specified experiment, and saves the results."""
    config_file = Path(config_path)
    base_dir = config_file.resolve().parent

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_config = config['experiment']
    output_config = config['output']
    
    output_dir = _resolve_path(base_dir, output_config['directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_config['results_file']
    
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
            def target_func(p, distribution, dynamics, backend='python', **run_settings):
                sim = simulator.SRCSimulator(
                    p=p,
                    distribution=distribution,
                    dynamics=dynamics,
                    backend=backend,
                )
                return sim.run_simulations(**run_settings)
            
            results = analysis.run_parameter_sweep(
                target_func=target_func, sweep_params=experiment_config['sweep_params'],
                fixed_params={
                    'distribution': dist,
                    'dynamics': dynamic_func,
                    'backend': experiment_config.get('backend', 'python'),
                    **experiment_config['run_settings'],
                }
            )
    
    elif task in ['fit_mle', 'fit_mcmc']:
        data_path = _resolve_path(base_dir, experiment_config['data_path'])
        with open(data_path, 'r') as f:
            data = [int(line.strip()) for line in f if line.strip()]
        
        inference = _load_inference_module()
        fitter = inference.Fitter(data, experiment_config)
        
        if task == 'fit_mle':
            results = fitter.fit_mle(**experiment_config['mle_settings'])
            results['optimizer_result'] = str(results['optimizer_result']) # Make serializable
            _save_results(results, output_path)
        
        elif task == 'fit_mcmc':
            results = fitter.fit_mcmc(**experiment_config['mcmc_settings'])
            results.to_netcdf(output_path)
    
    else:
        raise ValueError(f"Unknown task in config: {task}")

    if task in ['single_run', 'parameter_sweep']:
        _save_results(results, output_path)
    
    print(f"Results saved to {output_path}")

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
            az = _load_arviz()
            az.plot_trace(results)
            plt.tight_layout()
            plt.savefig(output_dir / "mcmc_trace.png")
            
            az.plot_posterior(results)
            plt.tight_layout()
            plt.savefig(output_dir / "mcmc_posterior.png")
        
        print(f"Plots saved in {output_dir}")
