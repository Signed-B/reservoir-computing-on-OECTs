import json
import shelve

dimindex = 1  # index in reservoir_dims of the dimension to plot

with shelve.open("Data/may8/dims_ensemble/data") as data:
    datadicts = data["dicts"]
    reservoir_dims = data["dims"]
    t = data["time"]

OECT_signal = datadicts[0]["OECT_signals"][dimindex]
OECT_prediction = datadicts[0]["OECT_predictions"][dimindex]
tanh_signal = datadicts[0]["tanh_signals"][dimindex]  # same as other signal
tanh_prediction = datadicts[0]["tanh_predictions"][dimindex]

d = {}
d["signal-x"] = [sig[0] for sig in OECT_signal]
d["signal-y"] = [sig[1] for sig in OECT_signal]
d["signal-z"] = [sig[2] for sig in OECT_signal]

d["OECT-prediction-x"] = [sig[0] for sig in OECT_prediction]
d["OECT-prediction-y"] = [sig[1] for sig in OECT_prediction]
d["OECT-prediction-z"] = [sig[2] for sig in OECT_prediction]

d["tanh-prediction-x"] = [sig[0] for sig in tanh_prediction]
d["tanh-prediction-y"] = [sig[1] for sig in tanh_prediction]
d["tanh-prediction-z"] = [sig[2] for sig in tanh_prediction]

d["t"] = t.tolist()

s = json.dumps(d)
with open("Data/timeseries_traces.json", "w") as file:
    file.write(s)
