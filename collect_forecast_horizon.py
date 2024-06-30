import json
import shelve

import numpy as np

from src import forecast_horizon

tol = 5  # Forecast horizon tolerance

filepath = [
    "Data/may8/dims_ensemble/data",
    "Data/may8/pinchoffs_ensemble/data",
    "Data/may8/sparse_ensemble/data",
    "Data/may8/alpha_ensemble/data",
]
variable_names = ["dims", "pinchoffs", "sparsities", "alphas"]
condensed_filenames = [
    "Data/FH_vs_size.json",
    "Data/FH_vs_pinchoff.json",
    "Data/FH_vs_p.json",
    "Data/FH_vs_alpha.json",
]

for i, f in enumerate(filepath):
    k = variable_names[i]
    new_f = condensed_filenames[i]

    with shelve.open(f) as data:
        datadicts = data["dicts"]
        v = np.array(data[k], dtype=float)
        t = data["time"]

    num_sims = len(datadicts)  # Number of simulations in run_dims.py
    num_params = len(v)

    FH_OECT = np.zeros((num_params, num_sims))
    FH_tanh = np.zeros((num_params, num_sims))

    # Ensemble begin
    for i in range(num_sims):
        OECT_signals = datadicts[i]["OECT_signals"]
        OECT_predictions = datadicts[i]["OECT_predictions"]
        if k != "pinchoffs":
            tanh_signals = datadicts[i]["tanh_signals"]
            tanh_predictions = datadicts[i]["tanh_predictions"]

        for j in range(num_params):
            OECT_signal = OECT_signals[j]
            OECT_prediction = OECT_predictions[j]

            tanh_signal = tanh_signals[j]
            tanh_prediction = tanh_predictions[j]

            FH_OECT[j, i] = forecast_horizon(OECT_signal, OECT_prediction, t, tol)
            if k != "pinchoffs":
                FH_tanh[j, i] = forecast_horizon(tanh_signal, tanh_prediction, t, tol)
    d = {}
    d["FH-OECT"] = FH_OECT.tolist()
    if k != "pinchoffs":
        d["FH-tanh"] = FH_tanh.tolist()
    d["tolerance"] = tol
    d["params"] = v.tolist()
    s = json.dumps(d)
    with open(new_f, "w") as file:
        file.write(s)
