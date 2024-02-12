from .utilities import get_output_layer
import numpy as np
import sys


def train_reservoir(n, D, u, A, ntraining, dt, w_in, alpha, function, tanshift, **args):
    r = np.zeros(n)

    X = np.zeros((ntraining, D))  # to store x,y,z Lorenz coordinates
    Z = np.zeros((ntraining, 2 * n))

    for t in range(ntraining):  # Training period

        Z[t] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)

        r = np.tanh(A.dot(r) + w_in.dot(u) + tanshift*np.ones(n))

        # sys.stdout.write("\rTraining " + str(t))

        X[t] = u  # store coordinates in X matrix

        # Could plot training period asw.
        # ye(t) = y # for plotting
        # ze(t) = z
        # xe(t) = x

    w_out = get_output_layer(Z, X, alpha)

    return w_out, u, r


def run_reservoir_autonomously(n, D, u, r, A, ntraining, ntesting, dt, w_in, w_out, function, tanshift, **args):
    # adding to Z matrix to hold self-feedback states.
    # Z = np.concatenate(Z, np.zeros(ntraining, n))
    Za = np.zeros((ntraining, 2 * n))

    v = np.dot(w_out, np.concatenate((r, np.square(r))))

    signal_during_auto = np.zeros((ntesting, D))
    pred_during_auto = np.zeros((ntesting, D))

    for t in range(ntesting):
        Za[t] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)

        r = np.tanh(A.dot(r) + w_in.dot(v) + tanshift*np.ones(n))

        v = np.dot(
            w_out, np.concatenate((r, np.square(r)))
        )  # get output using optimized output matrix w]

        # sys.stdout.write("\rTesting " + str(t))

        signal_during_auto[t] = u
        pred_during_auto[t] = v


    return signal_during_auto, pred_during_auto
