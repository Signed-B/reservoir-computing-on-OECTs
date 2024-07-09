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

    for id in range(n):
        if V1[id] - Vd[id] > Vp[id]:
            Vd[id] = Vdtemp1[id]
        # Cutoff regime
        elif V1[id] > Vp[id] and Vd[id] <= 0:
            Vd[id] = Vdtemp2[id]
        # Linear regime
        else:
            Vd[id] = Vdtemp3[id]


def train_oect_reservoir(
    u, tmax, dt, w_in, A, alpha, Vdinit, R, Rg, Cg, Vp, Kp, W, L, function, **args
):
    n = np.size(A, axis=0)

    # u = u0.copy()
    D = len(u)

    T = round(tmax / dt)

    V1 = np.zeros(n)
    Vd = np.zeros(n)

    X = np.zeros((T, D))  # to store x,y,z Lorenz coordinates
    r = np.zeros((T, n))
    r[0] = Vd

    for t in range(T - 1):  # Training period
        u += dt * function(u, t, **args)  # coordinates to feed to reservoir

        Vg = np.dot(w_in, u) + np.dot(A, Vd)  # TODO vector f, equation 14

        V1 = V1 + dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)
        r[t + 1] = Vd

        X[t] = u  # store coordinates in X matrix
    return get_output_layer(r, X, alpha), u, r[-1], V1


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

    r = np.zeros((T, n))
    r[0] = r0.copy()
    v = w_out.dot(r[0])

    Vd = r[0].copy()
    V1 = V1_0.copy()

    signal = np.zeros((T, D))
    prediction = np.zeros((T, D))

    t = np.zeros(T)
    t[0] = 0
    for i in range(T - 1):
        u += dt * function(u, t, **args)

        Vg = w_in.dot(v) + A.dot(Vd)  # TODO equation 21 (f vector)

        V1 += + dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)

        r[i + 1] = Vd.copy()

        v = w_out.dot(Vd)  # get output using optimized output matrix w]

        signal[i] = u
        prediction[i] = v

        t[i + 1] = t[i] + dt

    return t, signal, prediction
