"""
Evaluation metrics for denoising autoencoders.
Student 2 contribution: MSE and SNR computation and model evaluation pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Dict


def compute_mse(clean: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean Squared Error between clean and reconstructed signals."""
    return nn.functional.mse_loss(reconstructed, clean).item()


def compute_snr(clean: torch.Tensor, reconstructed: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Signal-to-Noise Ratio in dB.
    SNR = 10 * log10( ||x||^2 / ||x - x_hat||^2 )
    Higher is better.
    """
    # Sum over both channels (dim=-2) and time (dim=-1) to prevent low-power channels
    # from skewing the decibel average.
    signal_power = (clean ** 2).sum(dim=(-1, -2))          # (B,)
    noise_power  = ((clean - reconstructed) ** 2).sum(dim=(-1, -2))  # (B,)
    snr_per_sample = 10.0 * torch.log10(signal_power / (noise_power + eps))
    return snr_per_sample.mean().item()


def compute_input_snr(clean: torch.Tensor, noisy: torch.Tensor, eps: float = 1e-8) -> float:
    """SNR of the *noisy* input — baseline to compare against model output SNR."""
    return compute_snr(clean, noisy, eps)


def snr_improvement(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    reconstructed: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    SNR improvement (SNRi) = output_SNR - input_SNR.
    Measures how much the model improved over the noisy baseline.
    """
    return compute_snr(clean, reconstructed, eps) - compute_snr(clean, noisy, eps)


def compute_tremor_power_mae(clean: torch.Tensor, target: torch.Tensor, sample_rate: float = 100.0) -> float:
    """
    Computes Tremor Band Power (4-6 Hz) Mean Absolute Error.
    Useful for Parkinson's datasets to evaluate if the crucial frequency band is preserved.
    """
    signal_length = clean.shape[-1]
    freqs = torch.fft.rfftfreq(signal_length, d=1.0/sample_rate)
    
    # Indices for the 4-6 Hz band
    mask = (freqs >= 4.0) & (freqs <= 6.0)
    if not mask.any():
        return 0.0 # handle case where sampling rate/length doesn't cover band
        
    mask = mask.to(clean.device)
    
    # rfft not supported on MPS — always run on CPU (tensors already cpu here)
    clean_rfft  = torch.fft.rfft(clean.cpu(),  dim=-1, norm="ortho")
    target_rfft = torch.fft.rfft(target.cpu(), dim=-1, norm="ortho")
    
    # Power = abs(fft)^2
    clean_power = (torch.abs(clean_rfft)**2)[..., mask].sum(dim=-1)
    target_power = (torch.abs(target_rfft)**2)[..., mask].sum(dim=-1)
    
    mae = torch.abs(clean_power - target_power).mean()
    return mae.item()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    noise_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Run full evaluation over a DataLoader.
    Returns dict with keys: mse, snr_out, snr_in, snr_improvement.
    """
    model.eval()
    all_clean, all_noisy, all_recon = [], [], []

    for batch in loader:
        clean = batch.to(device)
        noisy = noise_fn(clean)
        recon = model(noisy)
        all_clean.append(clean.cpu())
        all_noisy.append(noisy.cpu())
        all_recon.append(recon.cpu())

    clean = torch.cat(all_clean)
    noisy = torch.cat(all_noisy)
    recon = torch.cat(all_recon)

    return {
        "mse":             compute_mse(clean, recon),
        "snr_out":         compute_snr(clean, recon),
        "snr_in":          compute_snr(clean, noisy),
        "snr_improvement": snr_improvement(clean, noisy, recon),
        "tremor_mae_out":  compute_tremor_power_mae(clean, recon),
        "tremor_mae_in":   compute_tremor_power_mae(clean, noisy),
    }
