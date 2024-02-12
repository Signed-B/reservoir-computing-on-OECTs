import sys

import numpy as np

from .utilities import get_output_layer


def update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L):
    n = len(Vg)
    a = Kp * W * R / L
    b = R / (2 * Rg)

    Vdtemp1 = Vdinit + (a / 2) * (V1 - Vp) ** 2 + b * (Vg - V1)
    Vdtemp2 = Vdinit + b * (Vg - V1)
    delta = 2 * a * (Vdinit + b * (Vg - V1)) + (a * (V1 - Vp) - 1) ** 2
    Vdtemp3 = -(1 / a) + (V1 - Vp) + (1 / a) * np.sqrt(np.maximum(delta, np.zeros(n)))

    # TODO 99% correct, check w/ Juan

    for id in range(n):
        if V1[id] - Vd[id] > Vp[id]:
            Vd[id] = Vdtemp1[id]
        # Cutoff regime
        elif V1[id] > Vp[id] and Vd[id] <= 0:
            Vd[id] = Vdtemp2[id]
        # Linear regime
        else:
            Vd[id] = Vdtemp3[id]
    # for id in range(n):
    #     if Vd[id] < Vg[id] - Vp[id]:
    #         Vd[id] = Vdtemp1[id]
    #     else:
    #         Vd[id] = Vdtemp3[id]


def train_oect_reservoir(
    u0, tmax, dt, w_in, A, alpha, Vdinit, R, Rg, Cg, Vp, Kp, W, L, function, **args
):
    n = np.size(A, axis=0)

    u = u0.copy()
    D = len(u)

    T = round(tmax / dt)

    V1 = np.zeros(n)
    Vd = np.zeros(n)

    X = np.zeros((T, D))  # to store x,y,z Lorenz coordinates
    Z = np.zeros((T, 2 * n))
    r = Vd

    for t in range(T):  # Training period

        Z[t] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)  # coordinates to feed to reservoir

        Vg = np.dot(w_in, u) + np.dot(A, Vd)  # TODO dot?
        V1 = V1 + dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)

        r = Vd

        # sys.stdout.write("\rTraining " + str(t))

        X[t] = u  # store coordinates in X matrix
    return get_output_layer(Z, X, alpha), u, r, V1


def run_oect_reservoir_autonomously(
    u0,
    r0,
    V1_0,
    tmax,
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
    function,
    **args
):
    n = np.size(A, axis=0)

    u = u0.copy()
    D = len(u)
    T = round(tmax / dt)

    v = np.dot(w_out, np.concatenate((r0, np.square(r0))))
    r = r0.copy()
    Vd = r
    V1 = V1_0.copy()

    signal = np.zeros((T, D))
    prediction = np.zeros((T, D))

    Z = np.zeros((T, 2 * n))

    for t in range(T):
        Z[t] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)

        Vg = np.dot(w_in, v) + np.dot(
            A, Vd
        )  # here we feed v (previous output) instead of u
        V1 = V1 + dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)

        r = Vd

        v = w_out.dot(
            np.concatenate((r, np.square(r)))
        )  # get output using optimized output matrix w]

        # sys.stdout.write("\rTesting " + str(t))

        signal[t] = u
        prediction[t] = v

    return signal, prediction
