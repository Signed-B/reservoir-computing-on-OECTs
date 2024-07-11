import json
import os
import time
from copy import deepcopy

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, uniform

from src import *

def run_OECT_prediction(
    fname, 
    u0,
    n,
    p,
    r_dist,
    parameters,
    alpha,
    training_time,
    testing_time,
):
    dt = 0.01
    frac = 1
    w_in_sigma = 0.004

    D = len(u0)
    sigma = 10
    rho = 28
    beta = 8 / 3

    u0 = u0.copy()

    Vdinit, R, Rg, Cg, Vp, Kp, W, L = generate_OECT_parameters(n, parameters)
    
    A = erdos_renyi_network(n, p, r_dist)
    while nx.is_connected(nx.Graph(A)):
        A = erdos_renyi_network(n, p, r_dist)

    w_in = input_layer(D, n, w_in_sigma)

    w_out, u0, r0, V1_0 = train_oect_reservoir(
        u0,
        training_time,
        dt,
        frac,
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

    t, signal, prediction = run_oect_reservoir_autonomously(
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

    data = {}
    data["t"] = t.tolist()
    data["signal"] = signal.tolist()
    data["prediction"] = prediction.tolist()

    datastring = json.dumps(data)

    with open(fname, "w") as output_file:
        output_file.write(datastring)
    print("Single run completed!", flush=True)

def run_tanh_prediction(
    fname,
    u0,
    n,
    p,
    r_dist,
    alpha,
    training_time,
    testing_time,
):
    dt = 0.01
    frac = 1
    w_in_sigma = 0.004

    D = len(u0)
    sigma = 10
    rho = 28
    beta = 8 / 3

    u0 = u0.copy()

    A = erdos_renyi_network(n, p, r_dist)
    while nx.is_connected(nx.Graph(A)):
        A = erdos_renyi_network(n, p, r_dist)

    w_in = input_layer(D, n, w_in_sigma)

    ## train_reservoir
    w_out, u0, r = train_reservoir(
        u0,
        A,
        training_time,
        dt,
        frac,
        w_in,
        alpha,
        lorenz,
        0,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    ## Run reservoir autonomously.
    t, signal, prediction = run_reservoir_autonomously(
        u0,
        r,
        A,
        testing_time,
        dt,
        w_in,
        w_out,
        lorenz,
        0,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )
    data = {}
    data["t"] = t.tolist()
    data["signal"] = signal.tolist()
    data["prediction"] = prediction.tolist()

    datastring = json.dumps(data)

    with open(fname, "w") as output_file:
        output_file.write(datastring)
    print("Single run completed!", flush=True)


# delete all previous data files
data_dir = "Data/Pinchoffs"
os.makedirs(data_dir, exist_ok=True)

for f in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, f))

# number of available cores
n_processes = len(os.sched_getaffinity(0))
print(f"Running on {n_processes} cores", flush=True)

iterations = 2

n = 100
pinchoffs = [-.6, -.5, -.3, -.1, 0, .1, .3, .6]

training_time = 300
testing_time = 100
dt = 0.01

w_in_sigma = 0.004
alpha = 0.00001

gateR = 2.7e4
gateC = 8.98e-7

parameters = dict()
parameters["transconductance"] = {"mean": 0.582e-3, "stddev": 0.0582e-3}
parameters["channel-width"] = {"mean": 200e-6, "stddev": 0}
parameters["channel-length"] = {"mean": 101e-6, "stddev": 0}
parameters["pinchoff-voltage"] = {"mean": -0.6, "stddev": 0}  # pinch-off voltage
parameters["weighting-resistor"] = {"mean": 500, "stddev": 100}
parameters["gate-capacitance"] = {"mean": gateC, "stddev": 0.1 * gateC}
parameters["gate-resistance"] = {"mean": gateR, "stddev": 0.1 * gateR}
parameters["applied-drain-voltage"] = {"mean": -0.05, "stddev": 0}

def parameters_func(pinch):
    params = deepcopy(parameters)
    params["pinchoff-voltage"]["mean"] = pinch
    return params


# system
D = 3
r_dist = uniform(100, 500)
delta_dist = norm(scale=0.005)
p = 6 / n
sigma = 10
rho = 28
beta = 8 / 3

ensemble_results = []


# BEGIN ENSEMBLE RUNS
print("run_pinchoff.py: BEGIN ENSEMBLE RUNS")

initt = time.time()

u0 = generate_initial_conditions(
    iterations,
    [-7.4, -11.1, 20],
    delta_dist,
    5000,
    0.0001,
    lorenz,
    sigma=10,
    rho=28,
    beta=8 / 3,
)
print("Initial conditions generated!", flush=True)

arglist = []
for pinch in pinchoffs:
    for i in range(iterations):
        arglist.append(
            (
                f"{data_dir}/{pinch}_{i}_OECT.json",
                u0[i].copy(),
                n,
                p,
                r_dist,
                deepcopy(parameters_func(pinch)),
                alpha,
                training_time,
                testing_time,
            )
        )


Parallel(n_jobs=n_processes)(delayed(run_OECT_prediction)(*arg) for arg in arglist)


# Unnecessary due to no pinchoff change.

# arglist = []
# for pinch in pinchoffs:
#     for i in range(iterations):
#         arglist.append(
#             (
#                 f"{data_dir}/{pinch}_{i}_tanh.json",
#                 u0[i].copy(),
#                 n,
#                 p,
#                 r_dist,
#                 alpha,
#                 training_time,
#                 testing_time,
#             )
#         )

# Parallel(n_jobs=n_processes)(delayed(run_tanh_prediction)(*arg) for arg in arglist)



