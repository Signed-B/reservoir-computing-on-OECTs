import json

n = 25
r = 1

with open("Data/Dims/{n}_{r}_OECT.json") as oect_data:
    reservoir_dims = oect_data["n"]
    t = oect_data["t"]
    OECT_signal = oect_data["signal"]
    OECT_prediction = oect_data["prediction"]

with open("Data/Dims/{n}_{r}_tanh.json") as oect_data:
    reservoir_dims = oect_data["n"]
    t = oect_data["t"]
    tanh_signal = oect_data["signal"]
    tanh_prediction = oect_data["prediction"]

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
