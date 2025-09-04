from typing import Callable

IntensityDynamics = Callable[[int, bool], int] # Type alias: (intensity, is_receptive) -> new_intensity

def standard_intensity_dynamics(current_intensity: int, is_receptive: bool) -> int:
    """The k -> k +/- 1 dynamic from the original paper."""
    return current_intensity + 1 if is_receptive else current_intensity - 1

def saturating_intensity_dynamics(max_intensity: int) -> IntensityDynamics:
    """An example of a custom dynamic where intensity has a ceiling."""
    def dynamic_func(current_intensity: int, is_receptive: bool) -> int:
        if is_receptive:
            return min(current_intensity + 1, max_intensity)
        else:
            return current_intensity - 1
    dynamic_func.__name__ = f"saturating_{max_intensity}"
    return dynamic_func