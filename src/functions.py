"""This is a list of the ODE functions that we use in the reservoir prediction"""

import numpy as np


def lorenz(X, t, sigma=28, rho=10, beta=8 / 3):
    """The function governing the evolution of the Lorenz attractor

    Parameters
    ----------
    X : np.array
        The current state of the system
    t : float
        The time
    sigma : float, optional
        Lorenz parameter 1, by default 28
    rho : float, optional
        Lorenz parameter 2, by default 10
    beta : float, optional
        Lorenz parameter 3, by default 8/3

    Returns
    -------
    np.array
        The derivative of the Lorenz system
    """
    u, v, w = X
    up = -sigma * (u - v)
    vp = (rho - w) * u - v
    wp = -beta * w + u * v
    return np.array([up, vp, wp])
