import numpy as np

from .utilities import get_output_layer


def train_reservoir(u0, A, tmax, dt, frac, w_in, alpha, function, tanshift, **args):
    """Train a conventional reservoir computer

    Parameters
    ----------
    u0 : np.array
        Initial condition of the dynamical system
    A : np.array
        Adjacency matrix
    tmax : float
        Time for which the reservoir computer is trained
    dt : float > 0
        The time step to evolve the reservoir and dynamical system.
    frac : int between 0 and 1
        The fraction f to train the reservoir (discarding the beginning of the time series)
    w_in : np.array
        The input layer
    A : np.array
        A weighted matrix specifying the effective adjacency matrix.
    alpha : float
        The ridge regression parameter
    function : lambda function or function
        The ODE function governing the evolution of the dynamical system.
    tanshift : float
        How much to translate the tanh function

    Returns
    -------
    tuple
        w_out : np.array
            The output layer
        u : np.array
            the final state of the signal
        r : np.array
            the final state of the reservoir
    """
    n = A.shape[0]
    D = len(u0)
    T = round(tmax / dt)

    u = u0.copy()

    r = np.zeros(n)
    X = np.zeros((T, D))  # to store x,y,z Lorenz coordinates
    Z = np.zeros((T, 2 * n))
    X[0] = u

    for t in range(T - 1):  # Training period

        Z[t + 1] = np.concatenate((r, np.square(r)))  # reservoir states with trick

        u += dt * function(u, t, **args)

        r = np.tanh(A.dot(r) + w_in.dot(u) + tanshift * np.ones(n))

        X[t + 1] = u  # store coordinates in X matrix

    w_out = get_output_layer(Z[-round(frac * T) :], X[-round(frac * T) :], alpha)

    return w_out, u, r


def run_reservoir_autonomously(
    u0, r0, A, tmax, dt, w_in, w_out, function, tanshift, **args
):
    """Run a trained reservoir computer to predict time-series data

    Parameters
    ----------
    u0 : np.array
        Initial condition
    r0 : np.array
        Initial states of the reservoir (drain voltages)
    A : np.array
        The effective adjacency matrix
    tmax : float
        The time over which to predict the dynamical system
    dt : float > 0
        The time step to evolve the dynamical system and the OECT RC
    w_in : np.array
        The input layer
    w_out : np.array
        The output layer
    function : lambda function or function
        The ODE function governing the evolution of the dynamical system.
    tanshift : float
        How much to translate the tanh function

    Returns
    -------
    tuple
        t : np.array
            The times at which the signal is output
        signal : np.array
            The ground truth signal
        prediction : np.array
            The prediction from the OECT RC
    """
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
