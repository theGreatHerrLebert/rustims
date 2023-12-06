import numpy as np
from numba import jit
from typing import Callable
from numpy.typing import NDArray


@jit(nopython=True)
def smooth_step(x, up_start, up_end, k):
    """
    Smooth step function that transitions from 0 to 1 between a and b.

    Parameters:
    x (float or np.array): Input value(s).
    a (float): Start of the transition range.
    b (float): End of the transition range.
    k (float): Steepness of the transition.

    Returns:
    float or np.array: Output of the smooth step function.
    """
    m = (up_start + up_end) / 2  # Midpoint of the transition
    return 1 / (1 + np.exp(-k * (x - m)))


@jit(nopython=True)
def smooth_step_up_down(x, up_start, up_end, down_start, down_end, k):
    """
    Smooth step function that transitions from 0 to 1 between a and b,
    and then from 1 back to 0 between c and d.

    Parameters:
    x (float or np.array): Input value(s).
    a (float): Start of the first transition range (0 to 1).
    b (float): End of the first transition range (0 to 1).
    c (float): Start of the second transition range (1 to 0).
    d (float): End of the second transition range (1 to 0).
    k (float): Steepness of the transitions.

    Returns:
    float or np.array: Output of the smooth step function.
    """
    return smooth_step(x, up_start, up_end, k) - smooth_step(x, down_start, down_end, k)


def ion_transition_function_midpoint(midpoint, window_length=36, k=15) -> Callable[[NDArray], NDArray]:
    """
    Returns a function that calculates ion detection efficiency for a given m/z value(s).

    Parameters:
    midpoint (float): Midpoint of the window on the m/z axis.
    window_length (float): Length of the window on the m/z axis.
    k (float): Steepness of the transitions.

    Returns:
    function: A function that takes m/z value(s) and returns ion detection efficiency.
    """
    half_window = window_length / 2
    quarter_window = window_length / 4

    up_start = midpoint - half_window
    up_end = midpoint - quarter_window
    down_start = midpoint + quarter_window
    down_end = midpoint + half_window

    def apply_transition(mz):
        """
        Applies ion transition function to m/z value(s).

        Parameters:
        mz (float or np.array): Input m/z value(s).

        Returns:
        float or np.array: Output representing ion detection efficiency.
        """
        return smooth_step_up_down(mz, up_start, up_end, down_start, down_end, k)

    return apply_transition
