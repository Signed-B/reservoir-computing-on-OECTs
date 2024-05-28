import shelve
import sys

import numpy as np
import scipy.sparse as sparse

from src import *
import time
from tenacity import retry, stop_after_attempt


class OECT:
    def __init__(self):
        self.training_time = 300 # training time/
        # training_time = 100
        self.testing_time = 100
        self.dt = 0.01

        self.ntraining = int(self.training_time / self.dt)
        self.ntesting = int(self.testing_time / self.dt)

        self.w_in_sigma = 0.004

        gateR = 2.7e4
        gateC = 8.98e-7

        self.parameters = dict()
        self.parameters["transconductance"] = {"mean": 0.582e-3, "stddev": 0.0582e-3}
        self.parameters["channel-width"] = {"mean": 200e-6, "stddev": 0}
        self.parameters["channel-length"] = {"mean": 101e-6, "stddev": 0}
        self.parameters["threshold-voltage"] = {"mean": -0.6, "stddev": 0} # pinch-off voltage
        self.parameters["weighting-resistor"] = {"mean": 500, "stddev": 100}
        self.parameters["gate-capacitance"] = {"mean": gateC, "stddev": 0.1 * gateC}
        self.parameters["gate-resistance"] = {"mean": gateR, "stddev": 0.1 * gateR}
        self.parameters["applied-drain-voltage"] = {"mean": -0.05, "stddev": 0}

        # system
        self.D = 3
        self.mu = 1.2
        pass

    @retry(stop=stop_after_attempt(10))
    def oect_iteration(self, u, dim=None, alpha=None, pinchoff=None, rewire=None):
        # Parameters
        if dim is None:
            dim = self.dim
        if alpha is None:
            alpha = self.alpha
        if rewire is None:
            rewire = self.rewire
        if pinchoff:
            self.parameters["threshold-voltage"]["mean"] = pinchoff

        n = dim
        u0 = u.copy()
        print("> Rewire", rewire)

        # OECT parameters
        Vdinit, R, Rg, Cg, Vp, Kp, W, L = generate_OECT_parameters(n, self.parameters)

        # A = sparse.rand(n, n, rewire).A # TODO: something / n instead?
        # A = A - np.diag(np.diag(A))
        # A = (mu / spectral_radius(A)) * A

        A = erdos_renyi_network(n, rewire, self.mu)

        w_in = self.w_in_sigma * (2.0 * np.random.rand(n, self.D) - np.ones((n, self.D)))

        w_out, u0, r0, V1_0 = train_oect_reservoir(
            u0,
            self.training_time,
            self.dt,
            w_in,
            A,
            alpha,
            Vdinit,
            R,
            Rg,
            Cg,
            Vp,
            Kp,
            W,
            L,
            lorenz,
            sigma=self.sigma,
            rho=self.rho,
            beta=self.beta,
        )

        # print("\n")

        signal, prediction = run_oect_reservoir_autonomously(
            u0,
            r0,
            V1_0,
            self.testing_time,
            self.dt,
            w_in,
            w_out,
            A,
            Vdinit,
            R,
            Rg,
            Cg,
            Vp,
            Kp,
            W,
            L,
            lorenz,
            sigma=self.sigma,
            rho=self.rho,
            beta=self.beta,
        )

        return signal, prediction