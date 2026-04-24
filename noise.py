"""
Noise models for generating corrupted signals.
Student 1 primary contribution; provided here for completeness.

All functions accept tensors of any shape (B, L) or (B, C, L) and return
the same shape.  The last dimension is always treated as the time axis.
"""

import torch
import random


def gaussian_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Additive Gaussian noise: x_noisy = x + N(0, sigma^2)."""
    return x + sigma * torch.randn_like(x)


def random_masking(
    x: torch.Tensor,
    mask_prob: float = 0.1,
    mask_len: int = 20,
) -> torch.Tensor:
    """
    Randomly zero-out contiguous segments along the time axis.
    mask_prob: probability that any given time index starts a masked segment.
    mask_len:  length of each masked segment.

    The same time-segment mask is applied across all channels so that the
    corruption simulates a complete sensor dropout (physically meaningful).
    """
    x_noisy = x.clone()
    B = x.size(0)
    L = x.size(-1)

    for b in range(B):
        i = 0
        while i < L:
            if random.random() < mask_prob:
                end = min(i + mask_len, L)
                # Use ... to slice the last dim regardless of ndim
                x_noisy[b, ..., i:end] = 0.0
                i = end
            else:
                i += 1
    return x_noisy


def impulse_noise(
    x: torch.Tensor,
    impulse_prob: float = 0.05,
    amplitude: float = 3.0,
) -> torch.Tensor:
    """Randomly insert spikes (outliers) into the signal."""
    x_noisy = x.clone()
    mask = torch.rand_like(x) < impulse_prob
    signs = torch.sign(torch.randn_like(x))
    x_noisy[mask] += amplitude * signs[mask]
    return x_noisy


def sinusoidal_interference(
    x: torch.Tensor,
    freq: float = 0.05,
    amplitude: float = 0.3,
) -> torch.Tensor:
    """Add a fixed-frequency sinusoidal interference signal."""
    L = x.shape[-1]
    t = torch.linspace(0, 1, L, device=x.device)
    interference = amplitude * torch.sin(2 * torch.pi * freq * L * t)
    # Broadcast (L,) across any leading dims (B,) or (B, C)
    return x + interference


def make_noise_fn(noise_type: str, **kwargs):
    """Return a noise function by name with fixed kwargs."""
    fns = {
        "gaussian":   gaussian_noise,
        "masking":    random_masking,
        "impulse":    impulse_noise,
        "sinusoidal": sinusoidal_interference,
    }
    if noise_type not in fns:
        raise ValueError(f"Unknown noise type '{noise_type}'. Choose from {list(fns)}.")
    fn = fns[noise_type]
    return lambda x: fn(x, **kwargs)
