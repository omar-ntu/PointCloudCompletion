from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _scatter(ax, pc: np.ndarray, title: str) -> None:
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=4)
    ax.set_title(title)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def plot_triplet(partial: np.ndarray, completed: np.ndarray, full: np.ndarray, path: str | Path) -> None:
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    _scatter(ax1, partial, "Partial input")
    _scatter(ax2, completed, "Completed output")
    _scatter(ax3, full, "Ground truth")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
