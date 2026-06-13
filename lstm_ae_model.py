"""LSTM autoencoder baseline for one-dimensional bearing windows."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LATENT_DIM, WINDOW_SIZE


class LSTMAE(nn.Module):
    """A compact sequence autoencoder used as a stricter journal baseline."""

    def __init__(
        self,
        input_size: int = WINDOW_SIZE,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 64,
        num_layers: int = 1,
        downsample_factor: int = 8,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.downsample_factor = downsample_factor
        self.seq_len = input_size // downsample_factor
        self.pool = nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        pooled = self.pool(x)
        seq = pooled.transpose(1, 2)
        _, (h_n, _) = self.encoder(seq)
        z = self.to_latent(h_n[-1])
        dec_seed = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(dec_seed)
        x_hat_low = self.output(dec_out).transpose(1, 2)
        x_hat = F.interpolate(
            x_hat_low,
            size=self.input_size,
            mode="linear",
            align_corners=False,
        )
        return x_hat, z, None

    def loss_function(self, x, x_hat, *_):
        recon = F.mse_loss(x_hat, x, reduction="mean") * self.input_size
        return recon, recon, torch.zeros((), device=x.device)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_hat, _, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=[1, 2])
