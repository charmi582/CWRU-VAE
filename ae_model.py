"""1-D convolutional autoencoder baseline for bearing anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LATENT_DIM, WINDOW_SIZE


class ConvAE(nn.Module):
    """Deterministic AE with the same convolutional backbone as the VAE."""

    def __init__(self, input_size: int = WINDOW_SIZE, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self._ch = 128
        self._len = input_size // 16
        self._flat = self._ch * self._len
        self.encoder_fc = nn.Sequential(
            nn.Linear(self._flat, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._flat),
            nn.ReLU(),
        )
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 7, stride=2, padding=3, output_padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x).flatten(1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, self._ch, self._len)
        return self.decoder_deconv(h)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, None

    def loss_function(self, x, x_hat, *_):
        recon = F.mse_loss(x_hat, x, reduction="mean") * self.input_size
        return recon, recon, torch.zeros((), device=x.device)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_hat, _, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=[1, 2])
