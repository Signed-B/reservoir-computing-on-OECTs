import random

import numpy as np
import scipy as sp
from numpy.linalg import norm as matrix_norm
from scipy import sparse
from scipy.stats import gamma
from sklearn.linear_model import Ridge


def input_layer(n, D, sigma):
    return sigma * np.random.uniform(low=-1, high=1, size=(D, n))


def erdos_renyi_network(n, p, dist, Rg=None):
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
    return np.divide(1, S * R)


def get_output_layer(r, signal, beta=0, solver="ridge"):
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


def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))


def generate_OECT_parameters(n, parameters):
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
    Vdinit = gamma_distribution(n, mean, stddev**2)

    return Vdinit, R, Rg, Cg, Vp, Kp, W, L


def gamma_distribution(n, mean, variance):
    if variance == 0:
        return mean * np.ones(n)
    else:
        alpha = mean**2 / variance
        beta = mean / variance
        return gamma(alpha, scale=1 / beta).rvs(n)


def forecast_horizon(signal, prediction, t, tol):
    i = np.argmax(matrix_norm(signal - prediction, axis=1, ord=2) > tol)
    return t[i]


def generate_initial_conditions(n, u0, dist, relaxation_time, dt, function, **args):
    D = len(u0)
    ics = np.zeros((n, D))
    for i in range(n):
        u = u0.copy() + dist.rvs(size=D)
        for t in range(relaxation_time):
            u += dt * function(u, t, **args)

        ics[i] = u

    return ics
