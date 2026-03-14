import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_import_src_cascades():
    import src_cascades

    assert src_cascades.__version__ == "3.0.0"
    assert callable(src_cascades.run_experiment_from_config)

def test_pyproject_declares_optional_extras():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert set(optional) >= {"inference", "cpp", "all"}
    assert "emcee>=3.1" in optional["inference"]
    assert "pybind11>=2.10" in optional["cpp"]

def test_fit_mcmc_requires_optional_dependencies():
    from src_cascades import Fitter

    fitter = Fitter(
        [1, 2, 3],
        {
            "distribution": {"name": "Poisson", "params": {"ell": 2.0}},
            "dynamics": {"name": "standard", "params": {}},
            "backend": "python",
        },
    )

    if importlib.util.find_spec("emcee") and importlib.util.find_spec("arviz"):
        result = fitter.fit_mcmc(n_walkers=4, n_steps=5, n_sim_per_step=5)
        assert hasattr(result, "posterior")
    else:
        with pytest.raises(ImportError, match="optional inference dependencies"):
            fitter.fit_mcmc(n_walkers=4, n_steps=5, n_sim_per_step=5)

def test_build_wheel_smoke(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-build-isolation",
            "--no-deps",
            "-w",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Successfully built" in completed.stdout
    assert list(tmp_path.glob("src_cascades-*.whl"))
