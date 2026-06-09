import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str, tight: bool = True, **kwargs) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if tight:
        fig.tight_layout()

    fig.savefig(path, bbox_inches='tight', **kwargs)
    plt.close(fig)
