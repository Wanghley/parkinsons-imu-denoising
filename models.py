import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Baseline: Fully-Connected (MLP) Autoencoder
# ──────────────────────────────────────────────
class FCAutoencoder(nn.Module):
    """
    Fully-connected encoder-decoder.
    Input/output shape: (B, num_channels, signal_length).
    Internally flattens to (B, num_channels * signal_length).
    """

    def __init__(
        self,
        signal_length: int = 1024,
        latent_dim: int = 64,
        num_channels: int = 1,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.num_channels = num_channels

        flat = signal_length * num_channels
        self.encoder = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, flat),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, C*L)
        return self.encoder(x.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # (B, latent) → (B, C, L)
        out = self.decoder(z)
        return out.view(out.size(0), self.num_channels, self.signal_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ──────────────────────────────────────────────
# Advanced: 1D Convolutional Autoencoder
# ──────────────────────────────────────────────
class CNNAutoencoder(nn.Module):
    """
    1D-CNN encoder-decoder.
    Input/output shape: (B, num_channels, signal_length).

    Encoder strides: L → L/2 → L/4 → L/8 → L/16 (×256 feature maps) → latent_dim
    Decoder reverses via ConvTranspose1d.
    """

    def __init__(
        self,
        signal_length: int = 1024,
        latent_dim: int = 64,
        num_channels: int = 1,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self._bottleneck_channels = 256
        self._bottleneck_len = signal_length // 16  # 4 strides of 2

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(num_channels, 32,  kernel_size=3, stride=2, padding=1),  # L/2
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(32,  64,  kernel_size=3, stride=2, padding=1),           # L/4
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(64,  128, kernel_size=3, stride=2, padding=1),           # L/8
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),           # L/16
            nn.ReLU(),
        )

        flat_size = self._bottleneck_channels * self._bottleneck_len
        self.encoder_fc = nn.Linear(flat_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # ×2
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.ConvTranspose1d(128, 64,  kernel_size=4, stride=2, padding=1),  # ×2
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.ConvTranspose1d(64,  32,  kernel_size=4, stride=2, padding=1),  # ×2
            nn.ReLU(),
            nn.ConvTranspose1d(32,  num_channels, kernel_size=4, stride=2, padding=1),  # ×2
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) — already channel-first, no unsqueeze needed
        h = self.encoder_conv(x)           # (B, 256, L/16)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)          # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self._bottleneck_channels, self._bottleneck_len)
        out = self.decoder_conv(h)         # (B, C, L) — no squeeze
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
    Input/output shape: (B, num_channels, signal_length).
    The signal is treated as a sequence of C-dimensional time-steps.

    Encoder: bidirectional LSTM over (B, L, C) → mean-pool → linear projection
    Decoder: repeat latent → LSTM → linear projection per step → (B, C, L)
    """

    def __init__(
        self,
        signal_length: int = 1024,
        latent_dim: int = 64,
        num_channels: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_size * 2, latent_dim)

        # Project latent to (num_layers, hidden_size) for h0 and c0
        self.decoder_h0 = nn.Linear(latent_dim, num_layers * hidden_size)
        self.decoder_c0 = nn.Linear(latent_dim, num_layers * hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_size, num_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, L, C) for LSTM
        seq = x.permute(0, 2, 1)
        _, (h_n, _) = self.encoder_lstm(seq)   # h_n: (2*layers, B, H) bidirectional
        # Concatenate final forward and backward hidden states of top layer
        h_fwd = h_n[-2]   # (B, H) — last layer forward
        h_bwd = h_n[-1]   # (B, H) — last layer backward
        ctx = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*H)
        return self.encoder_fc(ctx)              # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        # Build initial hidden/cell states from latent vector
        h0 = self.decoder_h0(z).view(B, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = self.decoder_c0(z).view(B, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        # Teacher-forcing style: feed zeros as input, let hidden state carry info
        inp = torch.zeros(B, self.signal_length, self.num_channels, device=z.device)
        out, _ = self.decoder_lstm(inp, (h0, c0))  # (B, L, H)
        # (B, L, C) → (B, C, L)
        return self.output_fc(out).permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ──────────────────────────────────────────────
# Advanced: 1D U-Net Autoencoder
# ──────────────────────────────────────────────
class UNet1D(nn.Module):
    """
    1D-UNet encoder-decoder with skip connections.
    Includes a linear bottleneck layer so the latent representation
    can be compared directly to the other architectures.
    """

    def __init__(
        self,
        signal_length: int = 1024,
        latent_dim: int = 64,
        num_channels: int = 1,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self._bottleneck_channels = 256
        self._bottleneck_len = signal_length // 16

        # Encoder blocks
        self.enc1 = nn.Sequential(nn.Conv1d(num_channels, 32, 3, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(32, 64, 3, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(64, 128, 3, 2, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv1d(128, 256, 3, 2, 1), nn.ReLU())

        flat_size = self._bottleneck_channels * self._bottleneck_len
        self.encoder_fc = nn.Linear(flat_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_size)

        # Decoder blocks (input channels doubled due to skip concatenation)
        self.dec4 = nn.Sequential(nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(128 + 128, 64, 4, 2, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(64 + 64, 32, 4, 2, 1), nn.ReLU())
        self.dec1 = nn.ConvTranspose1d(32 + 32, num_channels, 4, 2, 1)

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        h = e4.view(e4.size(0), -1)
        z = self.encoder_fc(h)
        return z, [e1, e2, e3, e4]

    def decode(self, z, skips):
        e1, e2, e3, _ = skips
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self._bottleneck_channels, self._bottleneck_len)

        d4 = self.dec4(h)
        d4 = self._pad_and_cat(d4, e3)
        d3 = self.dec3(d4)
        d3 = self._pad_and_cat(d3, e2)
        d2 = self.dec2(d3)
        d2 = self._pad_and_cat(d2, e1)
        out = self.dec1(d2)

        if out.size(-1) != self.signal_length:
            out = out[..., :self.signal_length]
        return out

    def _pad_and_cat(self, dec_tensor: torch.Tensor, enc_tensor: torch.Tensor) -> torch.Tensor:
        diff = enc_tensor.size(-1) - dec_tensor.size(-1)
        if diff > 0:
            dec_tensor = nn.functional.pad(dec_tensor, (diff // 2, diff - diff // 2))
        elif diff < 0:
            enc_tensor = nn.functional.pad(enc_tensor, (-diff // 2, -diff + diff // 2))
        return torch.cat([dec_tensor, enc_tensor], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, skips = self.encode(x)
        return self.decode(z, skips)


# ──────────────────────────────────────────────
# Optional: Patch-based Transformer Autoencoder
# ──────────────────────────────────────────────
class TransformerAutoencoder(nn.Module):
    """
    Patch-based 1D Transformer autoencoder (ViT-style tokenisation).

    The signal (B, C, L) is divided into non-overlapping patches of
    ``patch_size`` samples.  Each patch is linearly projected to a
    d_model-dimensional token.  With the default patch_size=8 and L=1024
    we get 128 tokens, keeping attention cost at O(128²) rather than O(1024²).

    Encoder : patch-embed → pos-embed → TransformerEncoder → mean-pool → FC → z
    Decoder : FC → tile to (B, P, d_model) → pos-embed → TransformerEncoder → patch-reconstruct
    """

    def __init__(
        self,
        signal_length: int = 1024,
        latent_dim: int = 64,
        num_channels: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        patch_size: int = 8,
    ):
        super().__init__()
        assert signal_length % patch_size == 0, \
            f"signal_length {signal_length} must be divisible by patch_size {patch_size}"

        self.signal_length = signal_length
        self.latent_dim    = latent_dim
        self.num_channels  = num_channels
        self.d_model       = d_model
        self.patch_size    = patch_size
        self.num_patches   = signal_length // patch_size  # 128 for L=1024, ps=8

        patch_dim = num_channels * patch_size  # e.g. 6 × 8 = 48

        # ---- Encoder ----
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers,
                                                     enable_nested_tensor=False)
        self.encoder_fc = nn.Linear(d_model, latent_dim)

        # ---- Decoder ----
        self.decoder_fc = nn.Linear(latent_dim, d_model)

        dec_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer_dec = nn.TransformerEncoder(dec_layer, num_layers,
                                                     enable_nested_tensor=False)
        self.patch_reconstruct = nn.Linear(d_model, patch_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        # (B, C, num_patches, patch_size) → (B, num_patches, C*patch_size)
        x = x.reshape(B, C, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, C * self.patch_size)

        tokens  = self.patch_embed(x) + self.pos_embed      # (B, P, d_model)
        enc_out = self.transformer_enc(tokens)               # (B, P, d_model)
        pooled  = enc_out.mean(dim=1)                        # (B, d_model)
        return self.encoder_fc(pooled)                       # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        # Tile latent to patch sequence, add positional info
        h = self.decoder_fc(z).unsqueeze(1).expand(-1, self.num_patches, -1)  # (B, P, d_model)
        h = h + self.pos_embed
        dec_out = self.transformer_dec(h)                    # (B, P, d_model)
        patches = self.patch_reconstruct(dec_out)            # (B, P, C*patch_size)

        # (B, P, C, patch_size) → (B, C, L)
        out = patches.reshape(B, self.num_patches, self.num_channels, self.patch_size)
        return out.permute(0, 2, 1, 3).reshape(B, self.num_channels, self.signal_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ──────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────
def build_model(
    name: str,
    signal_length: int = 1024,
    latent_dim: int = 64,
    num_channels: int = 1,
) -> nn.Module:
    """Return a model by name: 'fc', 'cnn', 'lstm', 'unet', or 'transformer'."""
    name = name.lower()
    if name == "fc":
        return FCAutoencoder(signal_length, latent_dim, num_channels)
    elif name == "cnn":
        return CNNAutoencoder(signal_length, latent_dim, num_channels)
    elif name == "lstm":
        return LSTMAutoencoder(signal_length, latent_dim, num_channels)
    elif name == "unet":
        return UNet1D(signal_length, latent_dim, num_channels)
    elif name == "transformer":
        return TransformerAutoencoder(signal_length, latent_dim, num_channels)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from 'fc', 'cnn', 'lstm', 'unet', 'transformer'.")
