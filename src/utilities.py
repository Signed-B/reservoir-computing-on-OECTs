import numpy as np
import scipy as sp
from scipy import sparse
from scipy.stats import gamma
from sklearn.linear_model import Ridge
import random


def erdos_renyi_network(
    n, p, spectral_radius, negative_weights=False, is_random_weight=True
):
    if is_random_weight and negative_weights:
        # weight = partial(random.uniform, -1, 1)
        weight = lambda: random.random() - 0.5
    elif is_random_weight and not negative_weights:
        weight = lambda: random.random()
    else:
        weight = lambda: 1
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if random.random() <= p and i != j:
                A[i, j] = weight()
    actual_spectral_radius = abs(np.max(np.linalg.eigvals(A)))
    return spectral_radius / actual_spectral_radius * A


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

    mean = parameters["threshold-voltage"]["mean"]
    stddev = parameters["threshold-voltage"]["stddev"]
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
