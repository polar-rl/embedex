import functools
import os
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def apply_latex_style[**P, R](build: Callable[P, R]) -> Callable[P, R]:
    """Serif text + Computer Modern math, so figures render in a LaTeX-like style."""

    @functools.wraps(build)
    def styled(*args: P.args, **kwargs: P.kwargs) -> R:
        with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'cm'}):
            return build(*args, **kwargs)

    return styled


def save_figure(fig: Figure, path: str, tight: bool = True, **kwargs) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if tight:
        fig.tight_layout()

    fig.savefig(path, bbox_inches='tight', **kwargs)
    plt.close(fig)
