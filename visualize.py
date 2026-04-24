"""
Visualization utilities.
Student 1 primary contribution; provided here for integration.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

_CHANNEL_NAMES = [
    "Accel X", "Accel Y", "Accel Z",
    "Gyro X",  "Gyro Y",  "Gyro Z",
]


def plot_signals(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Signal Reconstruction",
    save_path: Optional[str] = None,
    channel_names: Optional[List[str]] = None,
):
    """
    Plot clean, noisy, and reconstructed signals.

    Accepts either 1-D arrays of shape (L,) or multi-channel arrays of shape
    (C, L).  For multi-channel inputs each channel gets its own row; all three
    signal variants (clean / noisy / reconstructed) are overlaid within each row.
    """
    clean = np.asarray(clean)
    noisy = np.asarray(noisy)
    reconstructed = np.asarray(reconstructed)

    # Normalise to 2-D: (C, L)
    if clean.ndim == 1:
        clean         = clean[None]
        noisy         = noisy[None]
        reconstructed = reconstructed[None]

    C, L = clean.shape
    names = channel_names or (_CHANNEL_NAMES[:C] if C <= len(_CHANNEL_NAMES) else [f"Ch {i}" for i in range(C)])
    t = np.arange(L)

    fig, axes = plt.subplots(C, 1, figsize=(12, 2 * C), sharex=True)
    if C == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, clean[i],         color="steelblue", linewidth=0.8, label="clean")
        ax.plot(t, noisy[i],         color="orangered", linewidth=0.6, alpha=0.6, label="noisy")
        ax.plot(t, reconstructed[i], color="seagreen",  linewidth=0.8, alpha=0.85, label="recon")
        ax.set_ylabel(names[i], fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
):
    """Plot train/val loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train loss", color="steelblue")
    ax.plot(epochs, history["val_loss"],   label="Val loss",   color="orangered")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_latent_dim_results(
    latent_dims: List[int],
    mse_values: Dict[str, List[float]],
    snr_values: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot MSE and SNR vs. latent dimension for multiple models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    markers = ["o", "s", "^"]
    colors  = ["steelblue", "orangered", "seagreen"]

    for (model_name, mses), marker, color in zip(mse_values.items(), markers, colors):
        ax1.plot(latent_dims, mses, marker=marker, color=color, label=model_name)
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("MSE (↓)")
    ax1.set_title("MSE vs. Latent Dimension")
    ax1.set_xscale("log", base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for (model_name, snrs), marker, color in zip(snr_values.items(), markers, colors):
        ax2.plot(latent_dims, snrs, marker=marker, color=color, label=model_name)
    ax2.set_xlabel("Latent Dimension")
    ax2.set_ylabel("SNR (dB) (↑)")
    ax2.set_title("SNR vs. Latent Dimension")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_noise_robustness(
    noise_levels: List[float],
    snr_improvement: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot SNR improvement vs. noise level for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 4))
    markers = ["o", "s", "^"]
    colors  = ["steelblue", "orangered", "seagreen"]

    for (model_name, snri), marker, color in zip(snr_improvement.items(), markers, colors):
        ax.plot(noise_levels, snri, marker=marker, color=color, label=model_name)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="no improvement")
    ax.set_xlabel("Noise Level σ")
    ax.set_ylabel("SNR Improvement (dB)")
    ax.set_title("Noise Robustness: SNR Improvement vs. Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_noise_type_matrix(
    snri_matrix: Dict[str, Dict[str, float]],
    noise_types: List[str],
    arch: str,
    save_path: Optional[str] = None,
):
    """
    Heatmap of SNR improvement (dB) for every train-noise × test-noise combination.

    Rows = noise type used during training.
    Cols = noise type used during evaluation.
    Diagonal = matched condition; off-diagonal = cross-noise generalisation.
    """
    import numpy as np

    n = len(noise_types)
    matrix = np.array(
        [[snri_matrix[tr][te] for te in noise_types] for tr in noise_types],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(matrix.max()), abs(matrix.min()), 1)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="SNRi (dB)")

    labels = [t.capitalize() for t in noise_types]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Test noise type")
    ax.set_ylabel("Train noise type")
    ax.set_title(f"{arch.upper()} — Noise-Type Generalisation (SNRi dB)", fontweight="bold")

    for i in range(n):
        for j in range(n):
            color = "white" if abs(matrix[i, j]) > vmax * 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color=color,
                    fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_hyperparameter_search(
    table: List[Dict],
    lr_list: List[float],
    batch_list: List[int],
    save_path: Optional[str] = None,
):
    """
    Heatmap of best validation loss for every (learning_rate × batch_size) combination.
    Rows = learning rate, columns = batch size, color = best val loss (lower = better).
    """
    matrix = np.zeros((len(lr_list), len(batch_list)))
    for row in table:
        i = lr_list.index(row["lr"])
        j = batch_list.index(row["batch"])
        matrix[i, j] = row["best_val"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Best Val Loss (↓)")

    ax.set_xticks(range(len(batch_list)))
    ax.set_xticklabels([str(b) for b in batch_list], fontsize=9)
    ax.set_yticks(range(len(lr_list)))
    ax.set_yticklabels([f"{lr:.0e}" for lr in lr_list], fontsize=9)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Hyperparameter Search: Best Validation Loss", fontweight="bold")

    vmin, vmax = matrix.min(), matrix.max()
    for i in range(len(lr_list)):
        for j in range(len(batch_list)):
            color = "white" if matrix[i, j] < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
