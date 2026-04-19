"""
Visualization utilities.
Student 1 primary contribution; provided here for integration.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_signals(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Signal Reconstruction",
    save_path: Optional[str] = None,
):
    """Plot clean, noisy, and reconstructed 1D signals."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(clean))

    axes[0].plot(t, clean,         color="steelblue",  linewidth=0.8)
    axes[1].plot(t, noisy,         color="orangered",  linewidth=0.8, alpha=0.7)
    axes[2].plot(t, reconstructed, color="seagreen",   linewidth=0.8)
    axes[2].plot(t, clean,         color="steelblue",  linewidth=0.5, alpha=0.4, linestyle="--", label="clean (ref)")

    axes[0].set_ylabel("Clean")
    axes[1].set_ylabel("Noisy")
    axes[2].set_ylabel("Reconstructed")
    axes[2].set_xlabel("Sample index")
    axes[2].legend(fontsize=8)
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


def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
