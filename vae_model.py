"""1-D Convolutional Variational Autoencoder for bearing anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import WINDOW_SIZE, LATENT_DIM, BETA


class Encoder(nn.Module):
    """
    Input  : (B, 1, 1024)
    Output : mu (B, latent_dim), log_var (B, latent_dim)

    Architecture
    ────────────
    Conv1d: 1→16,  k=7, s=2, p=3  →  (16, 512)
    Conv1d: 16→32, k=5, s=2, p=2  →  (32, 256)
    Conv1d: 32→64, k=3, s=2, p=1  →  (64, 128)
    Conv1d: 64→128,k=3, s=2, p=1  →  (128, 64)
    Flatten                        →  8 192
    FC(8192→256) → FC(256→latent)  ×2  (mu, log_var)
    """

    def __init__(self, input_size: int = WINDOW_SIZE, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16,  7, stride=2, padding=3), nn.BatchNorm1d(16),  nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self._flat = 128 * (input_size // 16)
        self.fc      = nn.Sequential(nn.Linear(self._flat, 256), nn.ReLU())
        self.fc_mu   = nn.Linear(256, latent_dim)
        self.fc_logv = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).flatten(1)
        h = self.fc(h)
        return self.fc_mu(h), self.fc_logv(h)


class Decoder(nn.Module):
    """
    Input  : z  (B, latent_dim)
    Output : x̂  (B, 1, 1024)
    """

    def __init__(self, input_size: int = WINDOW_SIZE, latent_dim: int = LATENT_DIM):
        super().__init__()
        self._ch  = 128
        self._len = input_size // 16       # 64
        self._flat = self._ch * self._len  # 8 192

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, self._flat),  nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),  nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),  nn.ReLU(),
            nn.ConvTranspose1d(16,  1, 7, stride=2, padding=3, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self._ch, self._len)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Variational Autoencoder trained on **normal-only** data.

    Loss
    ────
    L = MSE(x, x̂) × input_size   (reconstruction)
      + β · KL( q(z|x) ‖ N(0,I) )  (regularisation)

    Anomaly score = per-sample reconstruction MSE.
    High error → anomaly (signal outside the learned normal manifold).
    """

    def __init__(
        self,
        input_size: int  = WINDOW_SIZE,
        latent_dim: int  = LATENT_DIM,
        beta: float      = BETA,
    ):
        super().__init__()
        self.encoder    = Encoder(input_size, latent_dim)
        self.decoder    = Decoder(input_size, latent_dim)
        self.input_size = input_size
        self.beta       = beta

    # ── reparameterisation ────────────────────────────────────────────────
    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu   # deterministic at inference

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decoder(z)
        return x_hat, mu, logvar

    # ── loss ──────────────────────────────────────────────────────────────
    def loss_function(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction="mean") * self.input_size
        kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        total = recon + self.beta * kl
        return total, recon, kl

    # ── anomaly scoring ───────────────────────────────────────────────────
    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE using deterministic (mean) latent code."""
        self.eval()
        mu, _ = self.encoder(x)
        x_hat  = self.decoder(mu)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=[1, 2])
