import matplotlib as mpl
import numpy as np

PLOT_RC = {
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
}


def apply_plot_style() -> None:
    mpl.rcParams.update(PLOT_RC)


CELL_DISPLAY = {
    'pointwise_fm': 'Pointwise (FM)',
    'pointwise_nf': 'Pointwise (NF)',
    'set_transformer_fm': 'Set Transformer (FM)',
    'set_transformer_nf': 'Set Transformer (NF)',
    'kde': 'KDE',
    'oracle': 'Oracle',
}


def mean_se(values) -> tuple[float, float]:
    """Mean and standard error over seeds (ddof=1; SE=0 for a single value)."""
    arr = np.array(values, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), se
