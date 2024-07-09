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
    u0, r, A, tmax, dt, w_in, w_out, function, tanshift, **args
):
    n = A.shape[0]
    D = len(u0)
    # adding to Z matrix to hold self-feedback states.
    # Z = np.concatenate(Z, np.zeros(ntraining, n))

    T = round(tmax / dt)
    Za = np.zeros((T, 2 * n))

    v = np.dot(w_out, np.concatenate((r, np.square(r))))

    signal = np.zeros((T, D))
    prediction = np.zeros((T, D))

    for t in range(T):
        Za[t] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)

        r = np.tanh(A.dot(r) + w_in.dot(v) + tanshift * np.ones(n))

        v = np.dot(
            w_out, np.concatenate((r, np.square(r)))
        )  # get output using optimized output matrix w]

        signal[t] = u
        prediction[t] = v

    return signal, prediction
