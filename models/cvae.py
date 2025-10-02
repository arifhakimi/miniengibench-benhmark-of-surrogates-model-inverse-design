"""Conditional variational autoencoder for design generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class CVAEConfig:
    """Lightweight configuration container for the conditional VAE."""

    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 400
    beta: float = 1.0
    device: str | None = None


class ConditionalVAE(nn.Module):
    """Conditional VAE that models p(x | y) for inverse design tasks."""

    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        config: CVAEConfig | None = None,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.cond_dim = int(cond_dim)
        self.config = config or CVAEConfig()
        self.latent_dim = int(self.config.latent_dim)
        self.hidden_dims = tuple(self.config.hidden_dims)
        self.beta = float(self.config.beta)
        device_name = self.config.device or (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = torch.device(device_name)

        encoder_layers: list[nn.Module] = []
        encoder_dims = [self.x_dim + self.cond_dim] + list(self.hidden_dims)
        for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # drop last ReLU
        self.encoder_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.encoder_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        decoder_layers: list[nn.Module] = []
        decoder_dims = [self.latent_dim + self.cond_dim] + list(self.hidden_dims)
        for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(self.hidden_dims[-1], self.x_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(self.device)
        self.training_history_: list[float] = []

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([x, y], dim=-1)
        hidden = self.encoder(inputs)
        mu = self.encoder_mu(hidden)
        logvar = self.encoder_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([z, y], dim=-1)
        return self.decoder(inputs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConditionalVAE":
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            drop_last=False,
        )

        opt = torch.optim.Adam(self.parameters(), lr=float(self.config.lr))
        self.training_history_.clear()

        for _ in range(int(self.config.epochs)):
            self.train()
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                opt.zero_grad()
                recon, mu, logvar = self.forward(batch_x, batch_y)
                recon_loss = F.mse_loss(recon, batch_x, reduction="mean")
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl
                loss.backward()
                opt.step()
                total_loss += loss.detach().cpu().item() * batch_x.size(0)
            epoch_loss = total_loss / len(dataset)
            self.training_history_.append(epoch_loss)

        return self

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.eval()
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        recon, _, _ = self.forward(X_tensor, y_tensor)
        return recon.cpu().numpy()

    @torch.no_grad()
    def generate(
        self,
        targets: Sequence[float] | np.ndarray,
        num_samples: int,
        z: Iterable[np.ndarray] | None = None,
    ) -> np.ndarray:
        self.eval()
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        if target_tensor.ndim == 1:
            target_tensor = target_tensor.unsqueeze(-1)
        if target_tensor.shape[0] != num_samples:
            if target_tensor.shape[0] == 1:
                target_tensor = target_tensor.repeat(num_samples, 1)
            else:
                raise ValueError(
                    "targets length must equal num_samples or be a single condition"
                )

        if z is not None:
            z_stack = np.stack(list(z), axis=0)
            if z_stack.shape[0] != num_samples:
                raise ValueError("Provided latent samples must match num_samples.")
            z_tensor = torch.as_tensor(z_stack, dtype=torch.float32, device=self.device)
        else:
            z_tensor = torch.randn((num_samples, self.latent_dim), device=self.device)

        samples = self.decode(z_tensor, target_tensor)
        return samples.cpu().numpy()

    def sample_prior(self, num_samples: int) -> np.ndarray:
        z = torch.randn((int(num_samples), self.latent_dim), device=self.device)
        return z.cpu().numpy()
