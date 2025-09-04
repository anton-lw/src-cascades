import numpy as np
import warnings
from .distributions import BranchingDistribution

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in power")

class PGFSolver:
    """
    Numerically solves the PGF recursion for the SRC model on a branching process.
    """
    def __init__(self, p: float, distribution: BranchingDistribution):
        self.p = p
        self.distribution = distribution

    def _solve_recursion(self, x_values, k_max, tolerance, max_iter):
        """Internal method to run the iterative fixed-point solver."""
        H = np.ones((k_max + 2, len(x_values)), dtype=np.complex128)
        H[0, :] = 1.0

        for i in range(max_iter):
            H_old = H.copy()
            H[k_max + 1, :] = H[k_max, :]

            for k in range(1, k_max + 1):
                arg_G = self.p * H[k + 1, :] + (1 - self.p) * H[k - 1, :]
                H[k, :] = x_values * self.distribution.pgf(arg_G)
            
            error = np.max(np.abs(H - H_old))
            if error < tolerance:
                return H
        
        print(f"Warning: PGF solver did not converge within {max_iter} iterations. Final error: {error:.2e}")
        return H

    def get_size_distribution(self, max_size: int = 2000, k_max: int = 100,
                              tolerance: float = 1e-12, max_iter: int = 5000) -> np.ndarray:
        """
        Calculates the cascade size probability distribution P(s) using the iFFT trick.
        """
        N = max_size
        j = np.arange(N)
        x_fft = np.exp(2j * np.pi * j / N)
        
        H_solution = self._solve_recursion(x_fft, k_max, tolerance, max_iter)
        H1_on_circle = H_solution[1, :]
        probs = np.real(np.fft.ifft(H1_on_circle))
        return np.maximum(0, probs)

    def get_supercritical_prob(self, k_max=100, tolerance=1e-12, max_iter=5000) -> float:
        """Calculates the probability of an infinite cascade, 1 - H_1(1)."""
        H_solution = self._solve_recursion(np.array([1.0]), k_max, tolerance, max_iter)
        H1_at_1 = H_solution[1, 0]
        return 1.0 - np.real(H1_at_1)

    def get_expected_size(self, initial_intensity: int, k_max: int = 200) -> float:
        """
        Calculates the expected cascade size for a given initial intensity, assuming
        the standard k -> k +/- 1 dynamics. Based on Eq. (3) from the paper.
        """
        ell = self.distribution.mean()
        p = self.p
        
        if ell <= 1 and p < 1:
            return (ell**initial_intensity - 1) / (ell - 1) if ell != 1 else float(initial_intensity)

        discriminant = 1 - 4 * p * (1 - p) * ell**2
        if discriminant < 0:
            return np.inf

        term1 = 1 / (ell - 1) if ell != 1 else np.inf
        if ell == 1: return float(initial_intensity)

        term2_num = 1 - np.sqrt(discriminant)
        term2_den = 2 * p * ell
        
        k_eff = min(initial_intensity, k_max)
        term2 = (term2_num / term2_den)**k_eff
        
        return term1 * (1 - term2)