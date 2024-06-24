import shelve
import sys

import numpy as np
import scipy.sparse as sparse

from src import *
import time

output = './Data/may8/dims_ensemble'

iterations = 2

# reservoir_dims = [10, 25, 50, 100, 200, 300, 500, 1000]
reservoir_dims = [10, 25,50]

# training_time = 300 # training time/
training_time = 100
testing_time = 100
dt = 0.01

ntraining = int(training_time / dt)
ntesting = int(testing_time / dt)

w_in_sigma = 0.004
alpha = 0.00001

# alpha multiple runs,
# grid search optimization

gateR = 2.7e4
gateC = 8.98e-7

parameters = dict()
parameters["transconductance"] = {"mean": 0.582e-3, "stddev": 0.0582e-3}
parameters["channel-width"] = {"mean": 200e-6, "stddev": 0}
parameters["channel-length"] = {"mean": 101e-6, "stddev": 0}
parameters["threshold-voltage"] = {"mean": -0.6, "stddev": 0} # pinch-off voltage
parameters["weighting-resistor"] = {"mean": 500, "stddev": 100}
parameters["gate-capacitance"] = {"mean": gateC, "stddev": 0.1 * gateC}
parameters["gate-resistance"] = {"mean": gateR, "stddev": 0.1 * gateR}
parameters["applied-drain-voltage"] = {"mean": -0.05, "stddev": 0}

# system
D = 3
mu = 1.2


ensemble_results = []

# BEGIN ENSEMBLE RUNS
print("run_dims.py: BEGIN ENSEMBLE RUNS")

initt = time.time()

ics = [[-7.4, -11.1, 20] + np.random.normal(size=3) * 0.05 for _ in range(iterations)]



for iter in range(iterations):
    print(f"========== Iteration {iter}/{iterations} ==========")
    st = time.time()

    # ==== OECT ====
    # print("OECT data generation.")
    # parameters for lorenz
    sigma = 10
    rho = 28
    beta = 8 / 3


    x = ics[iter][0]
    y = ics[iter][1]
    z = ics[iter][2]
    u = [x, y, z]

    # relax to attractor
    for t in range(5000):
        u += dt * lorenz(u, t, sigma, rho, beta)

    OECT_signals = []
    OECT_predictions = []

    

    print("> Generating OECT data...")
    for n in reservoir_dims:
        u0 = u.copy()
        print("> Dim", n)

        # OECT parameters
        Vdinit, R, Rg, Cg, Vp, Kp, W, L = generate_OECT_parameters(n, parameters)

        A = sparse.rand(n, n, 6 / n).A
        A = A - np.diag(np.diag(A))
        A = (mu / spectral_radius(A)) * A

        w_in = w_in_sigma * (2.0 * np.random.rand(n, D) - np.ones((n, D)))

        w_out, u0, r0, V1_0 = train_oect_reservoir(
            u0,
            training_time,off
            dt,
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
            sigma=sigma,
            rho=rho,
            beta=beta,
        )

        # print("\n")

        signal, prediction = run_oect_reservoir_autonomously(
            u0,
            r0,
            V1_0,
            testing_time,
            dt,
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
            sigma=sigma,
            rho=rho,
            beta=beta,
        )

        OECT_signals.append(signal)
        OECT_predictions.append(prediction)


    # ==== tanh ====
    # print("Tanh data generation.")


    tanh_signals = []
    tanh_predictions = []


    tanshift = 0


    # # parameters for lorenz
    # sigma = 10
    # rho = 28
    # beta = 8 / 3


    # x = -7.4
    # y = -11.1
    # z = 20
    # u = [x, y, z]

    # # relax to attractor
    # for t in range(5000):
    #     u += dt * lorenz(u, t, sigma, rho, beta)

    

    print("> Generating tanh data...")
    for n in reservoir_dims:
        u0 = u.copy()
        print("> Dim", n)

        A = sparse.rand(n, n, 6 / n).A
        A = A - np.diag(np.diag(A))
        A = (mu / spectral_radius(A)) * A

        w_in = w_in_sigma * (2.0 * np.random.rand(n, D) - np.ones((n, D)))

        ## train_reservoir
        w_out, u0, r = train_reservoir(n, D, u0, A, ntraining, dt, w_in, alpha, lorenz, tanshift, sigma=sigma, rho=rho, beta=beta)

        ## Run reservoir autonomously.
        signal_during_auto, pred_during_auto = run_reservoir_autonomously(
            n, D, u0, r, A, ntraining, ntesting, dt, w_in, w_out,
            lorenz, tanshift, sigma=sigma, rho=rho, beta=beta)


        tanh_signals.append(signal_during_auto)
        tanh_predictions.append(pred_during_auto)

    en_result = {"OECT_signals": OECT_signals,
                 "OECT_predictions": OECT_predictions,
                 "tanh_signals": tanh_signals,
                 "tanh_predictions": tanh_predictions
                 }
    
    ensemble_results.append(en_result)

    print(f"> Time elapsed: {time.time() - st:.2f} s, total time: {time.time() - initt:.2f} s")



# END ENSEMBLE


with shelve.open(f"{output}/data") as data:
    data["dicts"] = ensemble_results
    data["time"] = np.arange(0, testing_time, dt)
    data["dims"] = reservoir_dims
