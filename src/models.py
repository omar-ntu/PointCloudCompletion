from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        features = self.point_mlp(pc)
        pooled = features.max(dim=1).values
        return self.global_mlp(pooled)


class PointDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, n_points: int = 512):
        super().__init__()
        self.n_points = n_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, n_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.size(0), self.n_points, 3)


class PointAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, n_points: int = 512):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = PointDecoder(latent_dim, n_points)

    def forward(self, partial: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(partial))


class PointVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, n_points: int = 512):
        super().__init__()
        self.encoder_base = PointNetEncoder(256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = PointDecoder(latent_dim, n_points)

    def encode(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_base(pc)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(pc)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / max(half - 1, 1)))
        angles = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalLatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int = 128, cond_dim: int = 128, time_dim: int = 64, hidden_dim: int = 384):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, cond, self.time(t)], dim=-1))


@dataclass
class DiffusionSchedule:
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def tensors(self, device: torch.device):
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars


def q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor) -> torch.Tensor:
    a_bar = alpha_bars[t].view(-1, 1)
    return torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * noise


@torch.no_grad()
def sample_conditional_ddpm(
    model: ConditionalLatentDenoiser,
    cond: torch.Tensor,
    latent_dim: int,
    schedule: DiffusionSchedule,
    device: torch.device,
) -> torch.Tensor:
    n = cond.size(0)
    betas, alphas, alpha_bars = schedule.tensors(device)
    z = torch.randn(n, latent_dim, device=device)
    for i in reversed(range(schedule.timesteps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        pred_noise = model(z, cond, t)
        beta = betas[i]
        alpha = alphas[i]
        alpha_bar = alpha_bars[i]
        z = (1 / torch.sqrt(alpha)) * (z - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise)
        if i > 0:
            z = z + torch.sqrt(beta) * torch.randn_like(z)
    return z
