import json
import os

import numpy as np
from joblib import Parallel, delayed

from src import forecast_horizon


def collect_parameters(dir):
    varlist = set()
    rlist = set()

    for f in os.listdir(dir):
        d = f.split(".json")[0].split("_")

        var = float(d[0])
        r = int(d[1])

        varlist.add(var)
        rlist.add(r)

    v_dict = {c: i for i, c in enumerate(sorted(varlist))}
    r_dict = {r: i for i, r in enumerate(sorted(rlist))}

    return v_dict, r_dict


def get_data(f, dir, v_dict, r_dict, tol):
    fname = os.path.join(dir, f)
    d = f.split(".json")[0].split("_")
    var = float(d[0])
    r = int(d[1])
    rc = d[2]

    i = v_dict[var]
    j = r_dict[r]

    with open(fname, "r") as file:
        data = json.loads(file.read())
    t = np.array(data["t"], dtype=float)
    signal = np.array(data["signal"], dtype=float)
    prediction = np.array(data["prediction"], dtype=float)
    FH = forecast_horizon(signal, prediction, t, tol)
    return i, j, rc, FH


data_name = "dims"

data_dir = {
    "alpha": "Data/Alpha",
    "dims": "Data/Dims",
    "sparsity": "Data/Sparsity",
    "pinchoff": "Data/Pinchoffs",
}
var_name = {"alpha": "alpha", "dims": "n", "sparsity": "p", "pinchoff": "pinchoff"}
collected_fname = {
    "alpha": "Data/FH_vs_alpha.json",
    "dims": "Data/FH_vs_n.json",
    "sparsity": "Data/FH_vs_p.json",
    "pinchoff": "Data/FH_vs_pinchoff.json",
}

# get number of available cores
n_processes = len(os.sched_getaffinity(0))
print(f"Running on {n_processes} cores", flush=True)

# forecast horizon tolerance
tol = 5

v_dict, r_dict = collect_parameters(data_dir)

print(v_dict)
print(r_dict)

n_v = len(v_dict)
n_r = len(r_dict)

data = {}
data[var_name] = list(v_dict)
data["FH-OECT"] = np.zeros((n_v, n_r))
data["FH-tanh"] = np.zeros((n_v, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, v_dict, r_dict, tol))

results = Parallel(n_jobs=n_processes)(delayed(get_data)(*arg) for arg in arglist)

for i, j, rc, FH in results:
    if rc == "OECT":
        data["FH-OECT"][i, j] = FH
    elif rc == "tanh":
        data["FH-tanh"][i, j] = FH

for key, val in data.items():
    if not isinstance(val, list):
        data[key] = val.tolist()

datastring = json.dumps(data)

with open(collected_fname, "w") as output_file:
    output_file.write(datastring)
