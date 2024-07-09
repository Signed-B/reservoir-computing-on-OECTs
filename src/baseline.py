import numpy as np

from .utilities import get_output_layer


def train_reservoir(u0, A, tmax, dt, w_in, alpha, function, tanshift, **args):
    n = A.shape[0]
    D = len(u0)
    r = np.zeros(n)

    u = u0.copy()

    T = round(tmax / dt)

    X = np.zeros((T, D))  # to store x,y,z Lorenz coordinates
    Z = np.zeros((T, 2 * n))

    for t in range(T):  # Training period

        Z[t] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)

        r = np.tanh(A.dot(r) + w_in.dot(u) + tanshift * np.ones(n))

        X[t] = u  # store coordinates in X matrix

        # Could plot training period asw.
        # ye(t) = y # for plotting
        # ze(t) = z
        # xe(t) = x

    w_out = get_output_layer(Z, X, alpha)

    return w_out, u, r


def run_reservoir_autonomously(
    u0, r0, A, tmax, dt, w_in, w_out, function, tanshift, **args
):
    n = A.shape[0]
    D = len(u0)
    T = round(tmax / dt)

    signal = np.zeros((T, D))
    prediction = np.zeros((T, D))
    t = np.zeros(T)

    u = u0.copy()
    r = r0.copy()

    z = np.concatenate((r, np.square(r)))
    v = w_out.dot(z)

    signal[0] = u
    prediction[0] = v

    for i in range(T - 1):
        u += dt * function(u, t[i], **args)

        r = np.tanh(A.dot(r) + w_in.dot(v) + tanshift * np.ones(n))
        z = np.concatenate((r, np.square(r)))
        v = w_out.dot(z)  # get output using optimized output matrix w]

        signal[i + 1] = u
        prediction[i + 1] = v
        t[i + 1] = t[i] + dt

    return t, signal, prediction
