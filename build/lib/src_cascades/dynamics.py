from dataclasses import dataclass
from typing import Callable

IntensityDynamics = Callable[[int, bool], int] # Type alias: (intensity, is_receptive) -> new_intensity

def standard_intensity_dynamics(current_intensity: int, is_receptive: bool) -> int:
    """The k -> k +/- 1 dynamic from the original paper."""
    return current_intensity + 1 if is_receptive else current_intensity - 1

@dataclass(frozen=True)
class SaturatingIntensityDynamics:
    """Pickle-safe saturating dynamics for multiprocessing."""

    max_intensity: int

    def __call__(self, current_intensity: int, is_receptive: bool) -> int:
        if is_receptive:
            return min(current_intensity + 1, self.max_intensity)
        return current_intensity - 1

def saturating_intensity_dynamics(max_intensity: int) -> IntensityDynamics:
    """An example of a custom dynamic where intensity has a ceiling."""
    return SaturatingIntensityDynamics(max_intensity=max_intensity)
