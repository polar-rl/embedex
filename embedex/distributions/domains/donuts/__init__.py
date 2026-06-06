"""Donuts domain.

Bump matplotlib font sizes to ~1.5x the defaults for every donuts visualization.
Applied here so it takes effect on any import of a donuts module (training,
experiments, standalone plot scripts).
"""

import matplotlib as mpl

mpl.rcParams.update(
    {
        'font.size': 15,  # default 10
        'axes.titlesize': 18,  # default ~12 ('large')
        'axes.labelsize': 15,  # default 10 ('medium')
        'xtick.labelsize': 15,  # default 10 ('medium')
        'ytick.labelsize': 15,  # default 10 ('medium')
        'legend.fontsize': 15,  # default 10 ('medium')
        'figure.titlesize': 18,  # default ~14 ('large')
    }
)
