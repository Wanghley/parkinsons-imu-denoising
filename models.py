import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Baseline: Fully-Connected (MLP) Autoencoder
# ──────────────────────────────────────────────
class FCAutoencoder(nn.Module):
    """Fully-connected encoder-decoder. Input/output shape: (B, signal_length)."""

    def __init__(self, signal_length: int = 512, latent_dim: int = 64):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(signal_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, signal_length),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ──────────────────────────────────────────────
# Advanced: 1D Convolutional Autoencoder
# ──────────────────────────────────────────────
class CNNAutoencoder(nn.Module):
    """
    1D-CNN encoder-decoder.
    Input/output shape: (B, signal_length).
    Internally uses (B, 1, signal_length) for Conv1d layers.

    Encoder strides: 512 → 256 → 128 → 64 → 32 (×256 channels) → latent_dim
    Decoder reverses via ConvTranspose1d.
    """

    def __init__(self, signal_length: int = 512, latent_dim: int = 64):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self._bottleneck_channels = 256
        self._bottleneck_len = signal_length // 16  # 4 strides of 2

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),   # L/2
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # L/4
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # L/8
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),# L/16
            nn.ReLU(),
        )

        flat_size = self._bottleneck_channels * self._bottleneck_len
        self.encoder_fc = nn.Linear(flat_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, flat_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # ×2
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # ×2
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    # ×2
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),     # ×2
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x.unsqueeze(1))          # (B, 256, L/16)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)                       # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self._bottleneck_channels, self._bottleneck_len)
        out = self.decoder_conv(h).squeeze(1)           # (B, signal_length)
        # Trim or pad to exact signal_length in case of rounding
        if out.size(-1) != self.signal_length:
            out = out[..., : self.signal_length]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ──────────────────────────────────────────────
# Advanced (alt): LSTM Autoencoder
# ──────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    Input/output shape: (B, signal_length).
    The signal is treated as a sequence of scalar time-steps.

    Encoder: bidirectional LSTM → mean-pool hidden states → linear projection
    Decoder: repeat latent vector → LSTM → linear projection per step
    """

    def __init__(
        self,
        signal_length: int = 512,
        latent_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_fc = nn.Linear(hidden_size * 2, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_size, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) → (B, L, 1)
        seq = x.unsqueeze(-1)
        out, _ = self.encoder_lstm(seq)          # (B, L, 2*H)
        ctx = out.mean(dim=1)                    # (B, 2*H) — temporal mean pooling
        return self.encoder_fc(ctx)              # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Broadcast latent to sequence length
        h0 = self.decoder_fc(z)                 # (B, H)
        inp = h0.unsqueeze(1).expand(-1, self.signal_length, -1)  # (B, L, H)
        out, _ = self.decoder_lstm(inp)          # (B, L, H)
        return self.output_fc(out).squeeze(-1)   # (B, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ──────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────
def build_model(name: str, signal_length: int = 512, latent_dim: int = 64) -> nn.Module:
    """Return a model by name: 'fc', 'cnn', or 'lstm'."""
    name = name.lower()
    if name == "fc":
        return FCAutoencoder(signal_length, latent_dim)
    elif name == "cnn":
        return CNNAutoencoder(signal_length, latent_dim)
    elif name == "lstm":
        return LSTMAutoencoder(signal_length, latent_dim)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from 'fc', 'cnn', 'lstm'.")
