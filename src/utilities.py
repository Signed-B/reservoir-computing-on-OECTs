import random
import warnings

import numpy as np
import scipy as sp
from numpy.linalg import norm as matrix_norm
from scipy import sparse
from scipy.stats import gamma
from sklearn.linear_model import Ridge


def input_layer(n, D, sigma):
    """Generate the input layer

    Parameters
    ----------
    n : int
        Reservoir size
    D : int
        Dimension of the dynamical system
    sigma : float
        Max magnitude of the uniformly distributed coefficients

    Returns
    -------
    np.array
        The input layer
    """
    return sigma * np.random.uniform(low=-1, high=1, size=(D, n))


def erdos_renyi_network(n, p, dist, Rg=None):
    """Generates a weighted, directed Erdos-Renyi network

    Parameters
    ----------
    n : int
        Network size
    p : float between 0 and 1
        The connection probability
    dist : scipy.stats distribution
        The distribution of resistor values
    Rg : np.array, optional
        A vector specifying the gate resistances
        of all OECTs, by default None

    Returns
    -------
    A : np.array
        The effective adjacency matrix
    """
    # If it is not explicitly stated, we neglect the Rg term
    if Rg is None:
        Rg = np.inf * np.ones(n)

        # generate the theoretical directed network
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if random.random() <= p and i != j:
                    R[i, j] = dist.rvs()
                else:
                    R[i, j] = np.inf

    # create the adjacency matrix from the resistor network
    S = np.divide(1, Rg) + np.divide(1, R).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = np.divide(1, S * R)
    A[np.isnan(A)] = 0
    return A


def get_output_layer(r, signal, beta=0, solver="ridge"):
    """Solves for the output layer

    Parameters
    ----------
    r : np.array
        Reservoir states over time
    signal : np.array
        The ground truth signal
    beta : float, optional
        The ridge regression parameter, by default 0
    solver : str, optional
        Solver type, by default "ridge". Options include
        "ridge", "numpy", and "scipy".

    Returns
    -------
    output_layer : np.array
        The output layer
    """
    dim_reservoir = np.size(r, axis=1)

    if solver == "numpy":
        output_layer = np.linalg.solve(
            np.matmul(r.T, r) + beta * sparse.identity(dim_reservoir),
            np.matmul(r.T, signal),
        ).T
        return output_layer

    if solver == "scipy":
        output_layer = sp.linalg.solve(
            np.matmul(r.T, r) + beta * sparse.identity(dim_reservoir),
            np.matmul(r.T, signal),
        ).T
        return output_layer

    elif solver == "ridge":
        clf = Ridge(alpha=beta, solver="cholesky", fit_intercept=False)
        clf.fit(r, signal)
        return clf.coef_


def generate_OECT_parameters(n, parameters):
    """This generates arrays of OECT parameters

    Parameters
    ----------
    n : int
        Number of OECTs
    parameters : dict
        A dictionary describing the means and variances
        of all OECT parameters

    Returns
    -------
    tuple
        Vbias : np.array
            The bias voltage for each OECT
        R : np.array
            The weighting resistor from drain to ground
        Rg : np.array
            The effective resistance of the gate
        Cg : np.array
            The effective capacitance of the gate
        Vp : np.array
            The pinch-off voltage
        Kp : np.array
            The transconductance
        W : np.array
            The width of the OECT
        L : np.array
            The length of the OECT
    """
    mean = parameters["transconductance"]["mean"]
    stddev = parameters["transconductance"]["stddev"]
    Kp = gamma_distribution(n, mean, stddev**2)

    mean = parameters["channel-width"]["mean"]
    stddev = parameters["channel-width"]["stddev"]
    W = gamma_distribution(n, mean, stddev**2)

    mean = parameters["channel-length"]["mean"]
    stddev = parameters["channel-length"]["stddev"]
    L = gamma_distribution(n, mean, stddev**2)

    mean = parameters["pinchoff-voltage"]["mean"]
    stddev = parameters["pinchoff-voltage"]["stddev"]
    Vp = gamma_distribution(n, mean, stddev**2)

    mean = parameters["weighting-resistor"]["mean"]
    stddev = parameters["weighting-resistor"]["stddev"]
    R = gamma_distribution(n, mean, stddev**2)

    mean = parameters["gate-resistance"]["mean"]
    stddev = parameters["gate-resistance"]["stddev"]
    Rg = gamma_distribution(n, mean, stddev**2)

    mean = parameters["gate-capacitance"]["mean"]
    stddev = parameters["gate-capacitance"]["stddev"]
    Cg = gamma_distribution(n, mean, stddev**2)

    mean = parameters["applied-drain-voltage"]["mean"]
    stddev = parameters["applied-drain-voltage"]["stddev"]
    Vbias = gamma_distribution(n, mean, stddev**2)

    return Vbias, R, Rg, Cg, Vp, Kp, W, L


def gamma_distribution(n, mean, variance):
    """A gamma distribution which can handle zero
    variance.

    Parameters
    ----------
    n : int
        Number of values
    mean : float
        The mean of the distribution
    variance : float
        The variance of the distribution

    Returns
    -------
    float
        n samples from the gamma distribution.
    """
    if variance == 0:
        return mean * np.ones(n)
    else:
        alpha = mean**2 / variance
        beta = mean / variance
        return gamma(alpha, scale=1 / beta).rvs(n)


def forecast_horizon(signal, prediction, t, tol):
    """Calculating the forecast horizon

    Parameters
    ----------
    signal : np.array
        The time series to predict
    prediction : np.array
        The predicted time series
    t : np.array
        The corresponding times
    tol : float > 0
        The tolerance for the forecast horizon.

    Returns
    -------
    float
        The forecast horizon

    References
    ----------
    "Good and bad predictions: Assessing and improving the replication
    of chaotic attractors by means of reservoir computing" by
    Haluszczynski and Räth
    """
    i = np.argmax(matrix_norm(signal - prediction, axis=1, ord=2) > tol)
    return t[i]


def generate_initial_conditions(n, u0, dist, T, dt, function, **args):
    """Generate a set of initial condition on the attractor manifold

    Parameters
    ----------
    n : int
        Number of initial condition
    u0 : np.array
        The baseline initial condition
    dist : scipy.stats distribution
        The noise to add to the initial condition
    T : float
        The amount of time to allow the initial conditions to relax to the attractor
    dt : float > 0
        Time step
    function : lambda function or function
        The function describing the evolution of the dynamical system

    Returns
    -------
    np.array (n x D)
        An array of initial conditions.
    """
    D = len(u0)
    ics = np.zeros((n, D))
    for i in range(n):
        u = u0.copy() + dist.rvs(size=D)
        t = 0
        while t < T:
            u += dt * function(u, t, **args)
            t += dt

        ics[i] = u

    return ics
