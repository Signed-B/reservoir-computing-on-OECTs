<br />
<div align="center">
  <h3 align="center">A theoretical framework for reservoir computing on networks of of organic electrochemical transistors</h3>
  <h5 align="center">Nicholas W. Landry, Beckett R. Hyde, Jake C. Perez, Sean E. Shaheen, and Juan G. Restrepo</h5>
</div>

## Structure of the repository

* All the pre-processed data is contained in the [Data](/Data/) folder.
* Figures 3-8 presented in the paper are in the [Figures](/Figures/) folder.
* The standardized functions used in our analysis pipeline are in the [src](/src/) folder.

## Scripts

* The `run_alpha.py`, `run_dims.py`, `run_pinchoff.py`, and `run_sparsity.py` scripts train a standard RC and an OECT RC and use the trained network for prediction of the Lorenz attractor while varying the ridge regression parameter, the reservoir size, the pinch-off voltage, and the connection probability, respectively. These scripts generate the data for Figs. 8, 5, 6, and 7, respectively.
* The `collect_forecast_horizon.py` script consolidates the prediction data and calculates the forecast horizon for each realization and parameter value. The `collect_timeseries.py` script generates the consolidated data for Figs. 3 and 4.
* The `plot_forecast_horizon.ipynb` and `plot_timeseries.ipynb` notebooks generate all of the figures from the consolidated data.
* The `single_prediction.ipynb` notebook gives an example of a single prediction task for an OECT RC and conventional RC.

## Installation + Usage

To install the necessary requirements, clone this repository, navigate to the directory and run
```python
pip install -r requirements.txt
```

## License

Distributed under the BSD-3 License. See `LICENSE.txt` for more information.
