"""Plotting utilities for Mess3 transformer analysis."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _instance_colors(n):
    cmap = plt.cm.get_cmap("tab10", n)
    return [cmap(i) for i in range(n)]


def plot_training_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.loglog(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_instance_loss(instance_losses, instance_params):
    """Plot per-instance loss curves (log-log). instance_losses: dict[int, list[float]]."""
    n_inst = len(instance_params)
    colors = _instance_colors(n_inst)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) All on one plot
    ax = axes[0]
    for i in range(n_inst):
        epochs = np.arange(1, len(instance_losses[i]) + 1)
        a, x = instance_params[i]
        ax.loglog(epochs, instance_losses[i], color=colors[i], label=f"α={a}, x={x}", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Per-instance loss (log-log)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Each instance separately
    ax = axes[1]
    for i in range(n_inst):
        epochs = np.arange(1, len(instance_losses[i]) + 1)
        a, x = instance_params[i]
        ax.plot(epochs, instance_losses[i], color=colors[i], label=f"α={a}, x={x}", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Per-instance loss (linear)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_pca_per_instance(residuals, instance_ids, instance_params):
    """Shared-PCA scatter, one subplot per instance."""
    pca = PCA(n_components=2).fit(residuals)
    proj = pca.transform(residuals)
    n_inst = len(instance_params)
    colors = _instance_colors(n_inst)

    fig, axes = plt.subplots(1, n_inst, figsize=(3.5 * n_inst, 3.5))
    for i, ax in enumerate(axes):
        mask = instance_ids == i
        ax.scatter(proj[mask, 0], proj[mask, 1], s=3, alpha=0.4, color=colors[i])
        a, x = instance_params[i]
        ax.set_title(f"α={a}, x={x}", fontsize=9, color=colors[i])
        ax.set_xlabel("PC1")
        if i == 0: ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Residual stream PCA per instance (shared PCA, last position)", y=1.02)
    fig.tight_layout()
    return fig


def plot_cev_per_instance(residuals, instance_ids, instance_params):
    """Cumulative explained variance, one subplot per instance."""
    n_inst = len(instance_params)
    colors = _instance_colors(n_inst)
    fig, axes = plt.subplots(1, n_inst, figsize=(3.5 * n_inst, 3))
    for i, ax in enumerate(axes):
        mask = instance_ids == i
        cev_i = np.cumsum(PCA().fit(residuals[mask]).explained_variance_ratio_)
        ax.plot(range(1, len(cev_i) + 1), cev_i, "o-", markersize=2, color=colors[i])
        ax.axhline(0.95, color="r", linestyle="--", alpha=0.5)
        d95 = np.searchsorted(cev_i, 0.95) + 1
        ax.axvline(d95, color="g", linestyle="--", alpha=0.5, label=f"{d95}d @95%")
        a, x = instance_params[i]
        ax.set_title(f"α={a}, x={x}", fontsize=9, color=colors[i])
        ax.set_xlim(0, 30)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel("CEV")
    fig.suptitle("Cumulative explained variance per instance", y=1.02)
    fig.tight_layout()
    return fig


def plot_steps_x_instances_grid(all_projs, snap_steps, eval_ids, instance_params):
    """Grid: rows = instances, columns = training steps."""
    n_steps = len(snap_steps)
    n_inst = len(instance_params)
    colors = _instance_colors(n_inst)

    fig, axes = plt.subplots(n_inst, n_steps, figsize=(3 * n_steps, 3 * n_inst))
    for col, s in enumerate(snap_steps):
        proj = all_projs[s]
        for row in range(n_inst):
            ax = axes[row, col]
            mask = eval_ids == row
            ax.scatter(proj[mask, 0], proj[mask, 1], s=2, alpha=0.4, color=colors[row])
            if row == 0:
                ax.set_title(f"step {s}", fontsize=9)
            if col == 0:
                a, x = instance_params[row]
                ax.set_ylabel(f"α={a}, x={x}", fontsize=9, color=colors[row])
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Residual stream PCA: instances (rows) × training steps (cols)", y=1.01, fontsize=13)
    fig.tight_layout()
    return fig
