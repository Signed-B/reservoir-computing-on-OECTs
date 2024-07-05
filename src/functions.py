"""This is a list of the ODE functions that we use in the reservoir prediction"""

import numpy as np


def lorenz(X, t, sigma=28, rho=10, beta=8 / 3):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma * (u - v)
    vp = (rho - w) * u - v
    wp = -beta * w + u * v
    return np.array([up, vp, wp])


def rossler(X, t, a=0.1, b=0.1, c=14):
    u, v, w = X
    up = -(v + w)
    vp = u + a * v
    wp = b + w * (u - c)
    return np.array([up, vp, wp])
