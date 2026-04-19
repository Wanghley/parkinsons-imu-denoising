"""
Training and validation loop.
Student 1 primary contribution; provided here for integration.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Dict, List


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    noise_fn: Callable,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        clean = batch.to(device)
        noisy = noise_fn(clean)
        recon = model(noisy)
        loss  = nn.functional.mse_loss(recon, clean)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(clean)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    noise_fn: Callable,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        clean = batch.to(device)
        noisy = noise_fn(clean)
        recon = model(noisy)
        total_loss += nn.functional.mse_loss(recon, clean).item() * len(clean)
    return total_loss / len(loader.dataset)


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    noise_fn: Callable,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Full training loop with LR scheduling and optional checkpointing.
    Returns history dict with 'train_loss' and 'val_loss' lists.
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, noise_fn, device)
        val_loss   = validate(model, val_loader, noise_fn, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e}"
            )

    return history
