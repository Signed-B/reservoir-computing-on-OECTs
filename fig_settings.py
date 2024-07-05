# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:11:05 2021

@author: John Meluso
"""

import matplotlib as mpl
import matplotlib.pylab as pylab


def set_fonts(extra_params={}):
    params = {
        "font.family": "Serif",
        # "font.sans-serif": ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"],
        "mathtext.fontset": "cm",
        "legend.fontsize": 10,
        "axes.labelsize": 14,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 12,
    }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
