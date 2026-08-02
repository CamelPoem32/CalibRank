'''Small plotting helpers for validation notebooks.'''

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


##################################################
# Validation plot writers
##################################################
def save_singular_values_plot(values: ArrayLike, path: str | Path, title: str) -> Path:
    '''Save a singular-value stem plot.
    
    Args:
        values: Singular values, shape `(N,)`.
        path: Output image path.
        title: Figure title.
    
    Returns:
        pathlib.Path: Saved figure path.
    '''

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogy(np.arange(1, s.size + 1), np.maximum(s, 1e-16), marker="o")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_sparsity_plot(J: object, path: str | Path, title: str) -> Path:
    '''Save a matrix sparsity plot.
    
    Args:
        J: Dense or sparse matrix accepted by ``matplotlib.axes.Axes.spy``.
        path: Output image path.
        title: Figure title.
    
    Returns:
        Path to the saved image.
    '''

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.spy(J, markersize=2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out