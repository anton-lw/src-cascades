import importlib.util
import math

import pytest

from src_cascades.distributions import Poisson
from src_cascades.simulator import CPP_AVAILABLE, SRCSimulator

@pytest.mark.skipif(not CPP_AVAILABLE, reason="Optional C++ backend is not available.")
def test_cpp_backend_mean_matches_python_backend():
    python_result = SRCSimulator(p=0.05, distribution=Poisson(2.0), backend="python").run_simulations(
        num_simulations=2000,
        num_cores=1,
        initial_intensity=1,
    )
    cpp_result = SRCSimulator(p=0.05, distribution=Poisson(2.0), backend="cpp").run_simulations(
        num_simulations=2000,
        num_cores=1,
        initial_intensity=1,
    )

    assert math.isclose(python_result.mean_size, cpp_result.mean_size, rel_tol=0.2)

@pytest.mark.skipif(
    not (importlib.util.find_spec("emcee") and importlib.util.find_spec("arviz")),
    reason="Optional inference dependencies are not available.",
)
def test_fit_mcmc_returns_inference_data():
    from src_cascades import Fitter

    fitter = Fitter(
        [1, 2, 2, 3],
        {
            "distribution": {"name": "Poisson", "params": {"ell": 2.0}},
            "dynamics": {"name": "standard", "params": {}},
            "backend": "python",
        },
    )

    result = fitter.fit_mcmc(n_walkers=4, n_steps=5, n_sim_per_step=5)
    assert hasattr(result, "posterior")
