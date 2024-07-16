import numpy as np

from .utilities import get_output_layer


def update_drain_voltage(Vd, Vg, V1, Vdinit, R, Rg, Vp, Kp, W, L):
    """This function calculates the updated drain voltage.

    Updates occur in-place.

    Parameters
    ----------
    Vd : np.array
        The drain voltage of each OECT
    Vg : np.array
        The gate voltage of each OECT
    V1 : np.array
        V1 of each OECT
    Vdinit : np.array
        The bias voltage for each OECT
    R : np.array
        The weighting resistor from drain to ground
    Rg : np.array
        The effective resistance of the gate
    Vp : np.array
        The pinch-off voltage
    Kp : np.array
        The transconductance
    W : np.array
        The width of the OECT
    L : np.array
        The length of the OECT
    """
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
    """A script to train an OECT reservoir computer.

    Parameters
    ----------
    u0 : np.array
        The initial condition of the dynamical system
    tmax : float
        The time for which to train the reservoir computer
    dt : float > 0
        The time step to take with the reservoir and the ODE solver
    frac : int between 0 and 1
        The fraction f to train the reservoir (discarding the beginning of the time series)
    w_in : np.array
        The input layer
    A : np.array
        A weighted matrix specifying the effective adjacency matrix.
    alpha : float
        The ridge regression parameter
    Vdinit : np.array
        The bias voltage for each OECT
    R : np.array
        The weighting resistor from drain to ground
    Rg : np.array
        The effective resistance of the gate
    Cg : np.array
        The effective capacitance of the gate
    Vp : np.array
        The pinch-off voltage
    Kp : np.array
        The transconductance
    W : np.array
        The width of the OECT
    L : np.array
        The length of the OECT
    function : lambda function or function
        The ODE function governing the evolution of the dynamical system.

    Returns
    -------
    tuple
        w_out : np.array
            output layer
        u : np.array
            the final state of the dynamics which will be used as the initial
            condition for the prediction stage
        r[-1] : np.array
            the final reservoir states
        V1 : np.array
            the final values of V1 for each OECT

    Raises
    ------
    ValueError
        If there are NaNs in the training data
    """
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
    if np.any(np.isnan(r)):
        raise ValueError("There are NaN values in the reservoir states!")

    w_out = get_output_layer(r[-round(frac * T) :], X[-round(frac * T) :], alpha)
    return w_out, u, r[-1], V1


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
    """Predict time-series data using a trained OECT reservoir computer

    Parameters
    ----------
    u0 : np.array
        Initial condition
    r0 : np.array
        Initial states of the reservoir (drain voltages)
    V1_0 : np.array
        Initial values of V1 for every OECT
    tmax : float
        The time over which to predict the dynamical system
    dt : float > 0
        The time step to evolve the dynamical system and the OECT RC
    w_in : np.array
        The input layer
    w_out : np.array
        The output layer
    A : np.array
        The effective adjacency matrix
    Vdinit : np.array
        The bias voltage for each OECT
    R : np.array
        The weighting resistor from drain to ground
    Rg : np.array
        The effective resistance of the gate
    Cg : np.array
        The effective capacitance of the gate
    Vp : np.array
        The pinch-off voltage
    Kp : np.array
        The transconductance
    W : np.array
        The width of the OECT
    L : np.array
        The length of the OECT
    function : lambda function or function
        The ODE function governing the evolution of the dynamical system.

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
