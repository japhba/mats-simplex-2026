"""Plotting utilities for Mess3 transformer analysis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FIG_DIR = Path("figs")
FIG_DIR.mkdir(exist_ok=True)


def savefig(fig, name):
    """Save figure as 300 dpi PNG."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300)
    return path


def _instance_colors(n):
    cmap = plt.cm.get_cmap("tab10", n)
    return [cmap(i) for i in range(n)]


def _layer_names(n_layers):
    """Names for layers: 'emb', 'layer 0', ..., 'layer N-1'."""
    return ["emb"] + [f"layer {i}" for i in range(n_layers)]


def _layer_order(n):
    """Reversed: row 0 = deepest layer, row n-1 = embedding (bottom)."""
    return list(range(n - 1, -1, -1))


def embed_2d(data, method="pca"):
    """Project data to 2D via PCA."""
    if method == "pca":
        return PCA(n_components=2).fit_transform(data)
    raise ValueError(f"Unknown method: {method}")


def plot_training_loss(losses):
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.loglog(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    return fig


def plot_per_instance_loss(instance_losses, instance_params):
    """Plot per-instance loss curves (log-log + linear)."""
    n_inst = len(instance_params)
    colors = _instance_colors(n_inst)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')

    ax = axes[0]
    for i in range(n_inst):
        epochs = np.arange(1, len(instance_losses[i]) + 1)
        a, x = instance_params[i]
        ax.loglog(epochs, instance_losses[i], color=colors[i], label=f"α={a}, x={x}", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Per-instance loss (log-log)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(n_inst):
        epochs = np.arange(1, len(instance_losses[i]) + 1)
        a, x = instance_params[i]
        ax.plot(epochs, instance_losses[i], color=colors[i], label=f"α={a}, x={x}", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Per-instance loss (linear)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    return fig


def _project_layer(H, ids, beliefs, method, per_instance=True):
    """Project one layer's activations to 2D. Returns (N, seq_len, 2)."""
    N, seq_len, d_model = H.shape
    if method == "supervised":
        from lib import project_to_belief_2d
        return project_to_belief_2d(H, beliefs, ids, per_instance=per_instance)
    H_flat = H.reshape(-1, d_model)
    proj = embed_2d(H_flat, method=method)
    return proj.reshape(N, seq_len, 2)


def plot_embed_per_instance(H_layers, ids, instance_params, method="pca",
                            beliefs=None):
    """All-position embedding, rows=layers (bottom=emb), cols=instances."""
    n_layers = len(H_layers)
    n_inst = len(instance_params)
    colors = _instance_colors(n_inst)
    label = method.upper()
    layer_names = _layer_names(n_layers - 1)
    order = _layer_order(n_layers)

    fig, axes = plt.subplots(n_layers, n_inst, figsize=(3.5 * n_inst, 3.5 * n_layers), layout='constrained')
    for row_idx, layer_idx in enumerate(order):
        proj = _project_layer(H_layers[layer_idx], ids, beliefs, method, per_instance=True)
        for col in range(n_inst):
            ax = axes[row_idx, col]
            mask = ids == col
            P = proj[mask]
            ax.scatter(P[:, :, 0].ravel(), P[:, :, 1].ravel(), s=1, alpha=0.15, color=colors[col])
            if row_idx == 0:
                a, x = instance_params[col]
                ax.set_title(f"α={a}, x={x}", fontsize=9, color=colors[col])
            if col == 0:
                ax.set_ylabel(layer_names[layer_idx], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Residual stream {label} — all positions, per instance × layer", y=1.01, fontsize=13)
    return fig


def plot_embedding_vs_time(H_layers, ids, instance_params, method="pca",
                           positions=None, show_instances=None, title=None,
                           beliefs=None):
    """Instance separation over sequence position. Rows=layers (bottom=emb), cols=positions."""
    n_layers = len(H_layers)
    N, seq_len, d_model = H_layers[0].shape
    colors = _instance_colors(len(instance_params))
    label = method.upper()
    layer_names = _layer_names(n_layers - 1)
    order = _layer_order(n_layers)

    if show_instances is None:
        show_instances = [0, 1]
    if positions is None:
        positions = np.linspace(0, seq_len - 1, min(8, seq_len), dtype=int).tolist()
    n_pos = len(positions)

    inst_mask = np.isin(ids, show_instances)
    beliefs_sub = beliefs[inst_mask] if beliefs is not None else None

    fig, axes = plt.subplots(n_layers, n_pos, figsize=(3 * n_pos, 3 * n_layers), layout='constrained')
    for row_idx, layer_idx in enumerate(order):
        H_sub = H_layers[layer_idx][inst_mask]
        ids_sub = ids[inst_mask]
        # For separation: single readout across both instances
        proj = _project_layer(H_sub, ids_sub, beliefs_sub, method, per_instance=False)
        for col, pos in enumerate(positions):
            ax = axes[row_idx, col]
            for i in show_instances:
                mask = ids_sub == i
                ax.scatter(proj[mask, pos, 0], proj[mask, pos, 1], s=4, alpha=0.4, color=colors[i])
            if row_idx == 0: ax.set_title(f"pos {pos}", fontsize=9)
            if col == 0: ax.set_ylabel(layer_names[layer_idx], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    for i in show_instances:
        a, x = instance_params[i]
        axes[0, -1].scatter([], [], color=colors[i], label=f"α={a}, x={x}", s=20)
    axes[0, -1].legend(fontsize=5, markerscale=2, loc="upper right")

    if title is None:
        title = f"Instance separation: layers × positions — {label}"
    fig.suptitle(title, y=1.01, fontsize=13)
    return fig


def plot_training_dynamics_grid(all_layer_projs, snap_steps, eval_ids, instance_idx,
                                instance_params, n_layers):
    """Grid: rows=layers (bottom=emb), cols=training steps, single instance.

    all_layer_projs: dict[layer_idx][step] -> (N, 2) projections.
    """
    n_steps = len(snap_steps)
    color = _instance_colors(max(instance_idx + 1, 1))[instance_idx]
    layer_names = _layer_names(n_layers)
    order = _layer_order(n_layers + 1)  # +1 for emb
    n_rows = n_layers + 1
    a, x = instance_params[instance_idx]

    fig, axes = plt.subplots(n_rows, n_steps, figsize=(3 * n_steps, 3 * n_rows), layout='constrained')
    for row_idx, layer_idx in enumerate(order):
        for col, s in enumerate(snap_steps):
            ax = axes[row_idx, col]
            proj = all_layer_projs[layer_idx][s]
            mask = eval_ids == instance_idx
            ax.scatter(proj[mask, 0], proj[mask, 1], s=2, alpha=0.4, color=color)
            if row_idx == 0: ax.set_title(f"step {s}", fontsize=9)
            if col == 0: ax.set_ylabel(layer_names[layer_idx], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Training dynamics: instance α={a}, x={x}", y=1.01, fontsize=13)
    return fig


def plot_flow_fields(H_layers, tokens, ids, instance_idx, instance_params,
                     vocab_size=3, grid_res=30):
    """Flow field streamplot. Rows=layers (bottom=emb), cols=token identity."""
    from scipy.interpolate import griddata

    n_layers = len(H_layers)
    layer_names = _layer_names(n_layers - 1)
    token_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    mask = ids == instance_idx
    a, x = instance_params[instance_idx]
    order = _layer_order(n_layers)

    fig, axes = plt.subplots(n_layers, vocab_size, figsize=(5 * vocab_size, 5 * n_layers), layout='constrained')
    for row_idx, layer_idx in enumerate(order):
        H = H_layers[layer_idx][mask]
        toks = tokens[mask]
        n_i, seq_len, d_model = H.shape

        pca = PCA(n_components=2).fit(H.reshape(-1, d_model))
        proj = pca.transform(H.reshape(-1, d_model)).reshape(n_i, seq_len, 2)

        for col in range(vocab_size):
            ax = axes[row_idx, col]
            ax.scatter(proj[:, :, 0].ravel(), proj[:, :, 1].ravel(), s=0.5, alpha=0.05, color='grey')

            starts_list, dx_list = [], []
            for t in range(seq_len - 1):
                tok_mask = toks[:, t + 1] == col
                if not tok_mask.any():
                    continue
                starts_list.append(proj[tok_mask, t, :])
                dx_list.append(proj[tok_mask, t + 1, :] - proj[tok_mask, t, :])

            if starts_list:
                starts = np.concatenate(starts_list)
                dx = np.concatenate(dx_list)

                pad = 0.05
                xmin, xmax = starts[:, 0].min(), starts[:, 0].max()
                ymin, ymax = starts[:, 1].min(), starts[:, 1].max()
                xpad, ypad = (xmax - xmin) * pad, (ymax - ymin) * pad
                gx = np.linspace(xmin - xpad, xmax + xpad, grid_res)
                gy = np.linspace(ymin - ypad, ymax + ypad, grid_res)
                GX, GY = np.meshgrid(gx, gy)

                U = griddata(starts, dx[:, 0], (GX, GY), method='linear', fill_value=0)
                V = griddata(starts, dx[:, 1], (GX, GY), method='linear', fill_value=0)

                speed = np.sqrt(U**2 + V**2)
                lw = 1.5 * speed / (speed.max() + 1e-8)
                ax.streamplot(gx, gy, U, V, color=token_colors[col], linewidth=lw, density=1.5, arrowsize=1.2)

            if row_idx == 0: ax.set_title(f"$x_{{t+1}} = {col}$", fontsize=11, color=token_colors[col])
            if col == 0: ax.set_ylabel(layer_names[layer_idx], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Flow fields — instance α={a}, x={x}", y=1.01, fontsize=14)
    return fig
