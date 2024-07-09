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
    u0,
    tmax,
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
    function,
    **args
):
    n = A.shape[0]
    D = len(u0)
    T = round(tmax / dt)

    u = u0.copy()
    V1 = np.zeros(n)
    Vd = np.zeros(n)

    X = np.zeros((T, D))  # to store x,y,z Lorenz coordinates
    r = np.zeros((T, n))
    r[0] = Vd

    for t in range(T - 1):  # Training period
        u += dt * function(u, t, **args)  # coordinates to feed to reservoir

        Vg = w_in.dot(u) + A.dot(Vd)  # TODO vector f, equation 14

        V1 += dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)
        r[t + 1] = Vd

        X[t] = u  # store coordinates in X matrix
    return (
        get_output_layer(r[: round(frac * T)], X[: round(frac * T)], alpha),
        u,
        r[-1],
        V1,
    )


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
    D = len(u0)
    T = round(tmax / dt)

    u = u0.copy()
    signal = np.zeros((T, D))
    prediction = np.zeros((T, D))
    t = np.zeros(T)

    v = w_out.dot(r0)

    Vd = r0.copy()
    V1 = V1_0.copy()
    signal[0] = u.copy()
    prediction[0] = v

    for i in range(T - 1):
        u += dt * function(u, t, **args)

        Vg = w_in.dot(v) + A.dot(Vd)  # TODO equation 21 (f vector)

        V1 += dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)

        v = w_out.dot(Vd)  # get output using optimized output matrix w]

        signal[i + 1] = u
        prediction[i + 1] = v
        t[i + 1] = t[i] + dt

    return t, signal, prediction


def train_oect_reservoir_with_squaring(
    u0,
    tmax,
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
    function,
    **args
):
    n = A.shape[0]
    D = len(u0)
    T = round(tmax / dt)

    u = u0.copy()

    V1 = np.zeros(n)
    Vd = np.zeros(n)

    X = np.zeros((T, D))  # to store x,y,z Lorenz coordinates
    Z = np.zeros((T, 2 * n))
    r = Vd

    for t in range(T - 1):  # Training period

        Z[t] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)  # coordinates to feed to reservoir

        Vg = w_in.dot(u) + A.dot(Vd)  # TODO vector f, equation 14

        V1 = V1 + dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)

        r = Vd

        X[t] = u  # store coordinates in X matrix
    return get_output_layer(Z[: round(frac * T)], X[: round(frac * T)], alpha), u, r, V1


def run_oect_reservoir_autonomously_with_squaring(
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
    D = len(u0)
    T = round(tmax / dt)

    u = u0.copy()
    z = np.concatenate((r0, np.square(r0)))
    v = w_out.dot(z)
    r = r0.copy()
    Vd = r
    V1 = V1_0.copy()

    signal = np.zeros((T, D))
    prediction = np.zeros((T, D))
    t = np.zeros(T)

    signal[0] = u0
    prediction[0] = v

    for i in range(T - 1):
        u += dt * function(u, t[i], **args)

        Vg = w_in.dot(v) + A.dot(Vd)  # TODO equation 21 (f vector)

        V1 = V1 + dt * (Vg - V1) / (Rg * Cg)

        update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L)

        r = Vd

        z = np.concatenate((r, np.square(r)))
        v = w_out.dot(z)  # get output using optimized output matrix w]

        signal[i + 1] = u
        prediction[i + 1] = v
        t[i + 1] = t[i] + dt

    return t, signal, prediction
