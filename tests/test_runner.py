import json
from pathlib import Path

import pandas as pd
import yaml

from src_cascades.runner import run_experiment_from_config

def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path

def test_single_run_writes_json_results(tmp_path):
    config_path = _write_yaml(
        tmp_path / "single_run.yml",
        {
            "experiment": {
                "task": "single_run",
                "distribution": {"name": "FixedDegree", "params": {"k": 0}},
                "dynamics": {"name": "standard", "params": {}},
                "params": {"p": 0.2},
                "backend": "python",
                "run_settings": {
                    "num_simulations": 3,
                    "num_cores": 1,
                    "initial_intensity": 2,
                    "record_history": True,
                },
            },
            "output": {
                "directory": "out",
                "results_file": "single_run.json",
                "generate_plots": False,
            },
        },
    )

    run_experiment_from_config(str(config_path))

    payload = json.loads((tmp_path / "out" / "single_run.json").read_text(encoding="utf-8"))
    assert payload["summary"]["num_runs"] == 3
    assert len(payload["run_metrics"]) == 3
    assert all(metric["size"] == 1 for metric in payload["run_metrics"])

def test_parameter_sweep_writes_csv_and_uses_configured_dynamics(tmp_path):
    standard_config = {
        "experiment": {
            "task": "parameter_sweep",
            "distribution": {"name": "FixedDegree", "params": {"k": 2}},
            "dynamics": {"name": "standard", "params": {}},
            "backend": "python",
            "sweep_params": {"p": [0.2, 0.3]},
            "run_settings": {
                "num_simulations": 150,
                "num_cores": 1,
                "initial_intensity": 1,
                "max_steps": 200,
            },
        },
        "output": {
            "directory": "standard",
            "results_file": "sweep.csv",
            "generate_plots": False,
        },
    }
    saturating_config = {
        **standard_config,
        "experiment": {
            **standard_config["experiment"],
            "dynamics": {"name": "saturating", "params": {"max_intensity": 1}},
        },
        "output": {
            "directory": "saturating",
            "results_file": "sweep.csv",
            "generate_plots": False,
        },
    }

    run_experiment_from_config(str(_write_yaml(tmp_path / "standard.yml", standard_config)))
    run_experiment_from_config(str(_write_yaml(tmp_path / "saturating.yml", saturating_config)))

    standard_df = pd.read_csv(tmp_path / "standard" / "sweep.csv")
    saturating_df = pd.read_csv(tmp_path / "saturating" / "sweep.csv")

    assert list(standard_df["p"]) == [0.2, 0.3]
    assert not standard_df["mean_size"].equals(saturating_df["mean_size"])

def test_fit_mle_writes_results_json(tmp_path):
    data_path = tmp_path / "sample_data.txt"
    data_path.write_text("1\n2\n1\n3\n2\n", encoding="utf-8")
    config_path = _write_yaml(
        tmp_path / "fit_mle.yml",
        {
            "experiment": {
                "task": "fit_mle",
                "data_path": str(data_path.name),
                "distribution": {"name": "Poisson", "params": {"ell": 2.0}},
                "dynamics": {"name": "standard", "params": {}},
                "backend": "python",
                "mle_settings": {
                    "initial_guess": [0.05, 2.0],
                    "n_sim_per_step": 5,
                },
            },
            "output": {
                "directory": "fit",
                "results_file": "mle.json",
                "generate_plots": False,
            },
        },
    )

    run_experiment_from_config(str(config_path))

    payload = json.loads((tmp_path / "fit" / "mle.json").read_text(encoding="utf-8"))
    assert set(payload["mle_params"]) == {"p", "ell"}
    assert "AIC" in payload
    assert "BIC" in payload
