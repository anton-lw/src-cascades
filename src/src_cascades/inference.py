import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom
import emcee
import arviz as az
from typing import List, Dict, Any
from .simulator import SRCSimulator
from .distributions import Poisson

class Fitter:
    """A class to fit SRC model parameters to empirical cascade size data."""

    def __init__(self, data: List[int], config: Dict[str, Any]):
        """
        Args:
            data (List[int]): A list of observed cascade sizes.
            config (Dict): A configuration dictionary specifying the model to fit.
        """
        self.data = np.array(data)
        self.config = config
        self.simulator = self._create_simulator_from_config(config)

    def _create_simulator_from_config(self, config):
        dist = Poisson(ell=config.get('ell', 3.0))
        return SRCSimulator(
            p=config.get('p', 0.01),
            distribution=dist,
            backend=config.get('backend', 'cpp')
        )

    def _log_likelihood(self, params: np.ndarray, n_sim: int) -> float:
        """Calculate the log-likelihood of the data given model parameters."""
        p, ell = params
        if not (0 < p < 1 and 0 < ell < 15): # Priors/bounds
            return -np.inf

        self.simulator.p = p
        self.simulator.distribution.ell = ell
        
        sim_results = self.simulator.run_simulations(num_simulations=n_sim, num_cores=1)
        sim_sizes = sim_results.cascade_sizes
        
        if len(sim_sizes[sim_sizes > 0]) < 2: return -np.inf
        
        mean_sim = np.mean(sim_sizes)
        var_sim = np.var(sim_sizes)
        if var_sim <= mean_sim or mean_sim == 0: return -np.inf

        size_param = mean_sim**2 / (var_sim - mean_sim)
        prob_param = size_param / (size_param + mean_sim)
        
        log_lik = np.sum(nbinom.logpmf(self.data, n=size_param, p=prob_param))
        return log_lik if np.isfinite(log_lik) else -np.inf

    def fit_mle(self, initial_guess: List[float], n_sim_per_step: int = 1000) -> Dict:
        """Find the Maximum Likelihood Estimate (MLE) for model parameters."""
        neg_log_lik = lambda params: -self._log_likelihood(params, n_sim_per_step)
        
        result = minimize(
            neg_log_lik, x0=np.array(initial_guess),
            method='Nelder-Mead', options={'maxiter': 50, 'adaptive': True}
        )

        log_lik_final = -result.fun
        k = len(initial_guess)
        n = len(self.data)
        aic = 2 * k - 2 * log_lik_final
        bic = np.log(n) * k - 2 * log_lik_final

        return {
            "mle_params": {"p": result.x[0], "ell": result.x[1]},
            "log_likelihood": log_lik_final, "AIC": aic, "BIC": bic,
            "optimizer_result": result
        }

    def fit_mcmc(self, n_walkers: int, n_steps: int, n_sim_per_step: int = 500) -> az.InferenceData:
        """Fit model parameters using Markov Chain Monte Carlo (MCMC)."""
        log_prob_fn = lambda params: self._log_likelihood(params, n_sim_per_step)
        
        initial_state = np.random.rand(n_walkers, 2)
        initial_state[:, 1] *= 5 # Scale ell initial guess
        n_dims = initial_state.shape[1]

        sampler = emcee.EnsembleSampler(n_walkers, n_dims, log_prob_fn)
        sampler.run_mcmc(initial_state, n_steps, progress=True)
        
        inference_data = az.from_emcee(sampler)
        return inference_data