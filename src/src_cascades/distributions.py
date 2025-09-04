from typing import Protocol
import numpy as np

class BranchingDistribution(Protocol):
    """A protocol defining the interface for branching distributions."""
    def mean(self) -> float:
        """The mean of the distribution."""
        ...
    def sample(self) -> int:
        """Draw a single sample for the number of children."""
        ...
    def pgf(self, x: np.ndarray) -> np.ndarray:
        """The Probability Generating Function (PGF) of the distribution."""
        ...

class Poisson:
    def __init__(self, ell: float):
        self.ell = ell
        self.rng = np.random.default_rng()
    def mean(self) -> float:
        return self.ell
    def sample(self) -> int:
        return self.rng.poisson(self.ell)
    def pgf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.ell * (x - 1))

class FixedDegree:
    def __init__(self, k: int):
        self.k = k
    def mean(self) -> float:
        return float(self.k)
    def sample(self) -> int:
        return self.k
    def pgf(self, x: np.ndarray) -> np.ndarray:
        return x**self.k

class Geometric:
    def __init__(self, p: float):
        self.p = p # p is the success probability for the geometric distribution
        self.rng = np.random.default_rng()
    def mean(self) -> float:
        return (1 - self.p) / self.p
    def sample(self) -> int:
        return self.rng.geometric(self.p) - 1
    def pgf(self, x: np.ndarray) -> np.ndarray:
        return self.p / (1 - (1 - self.p) * x)