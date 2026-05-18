"""Small end-to-end smoke test for point-cloud completion.

This trains tiny models briefly on synthetic shapes and writes PLY/PNG outputs.
For meaningful metrics, run the full training commands in README.md.
"""

from __future__ import annotations

from pathlib import Path

import torch
torch.set_num_threads(2)
from torch.utils.data import DataLoader

from src.data import SyntheticCompletionDataset, save_ply, set_seed
from src.losses import chamfer_distance
from src.models import (
    ConditionalLatentDenoiser,
    DiffusionSchedule,
    PointNetEncoder,
    PointVAE,
    q_sample,
    sample_conditional_ddpm,
)
from src.visualize import plot_triplet


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticCompletionDataset(num_shapes=24, n_full=128, n_partial=64, seed=42)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    vae = PointVAE(latent_dim=16, n_points=128).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    for _ in range(1):
        for batch in loader:
            full = batch["full"].to(device)
            recon, mu, logvar, _ = vae(full)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = chamfer_distance(recon, full) + 0.001 * kld
            opt.zero_grad(); loss.backward(); opt.step()

    partial_encoder = PointNetEncoder(16).to(device)
    denoiser = ConditionalLatentDenoiser(latent_dim=16, cond_dim=16, hidden_dim=64).to(device)
    opt_d = torch.optim.AdamW(list(partial_encoder.parameters()) + list(denoiser.parameters()), lr=1e-3)
    schedule = DiffusionSchedule(timesteps=8)
    _, _, alpha_bars = schedule.tensors(device)
    for _ in range(1):
        for batch in loader:
            partial, full = batch["partial"].to(device), batch["full"].to(device)
            with torch.no_grad():
                z0, _ = vae.encode(full)
            cond = partial_encoder(partial)
            t = torch.randint(0, schedule.timesteps, (partial.size(0),), device=device)
            noise = torch.randn_like(z0)
            pred = denoiser(q_sample(z0, t, noise, alpha_bars), cond, t)
            loss = torch.nn.functional.mse_loss(pred, noise)
            opt_d.zero_grad(); loss.backward(); opt_d.step()

    sample = dataset[0]
    partial = sample["partial"].unsqueeze(0).to(device)
    full = sample["full"].unsqueeze(0).to(device)
    with torch.no_grad():
        cond = partial_encoder(partial)
        z = sample_conditional_ddpm(denoiser, cond, 16, schedule, device)
        completed = vae.decode(z)

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    save_ply(partial.squeeze(0).cpu().numpy(), out / "demo_partial.ply")
    save_ply(completed.squeeze(0).cpu().numpy(), out / "demo_completed.ply")
    save_ply(full.squeeze(0).cpu().numpy(), out / "demo_ground_truth.ply")
    plot_triplet(partial.squeeze(0).cpu().numpy(), completed.squeeze(0).cpu().numpy(), full.squeeze(0).cpu().numpy(), out / "demo_completion_triplet.png")
    print("Demo complete. See outputs/demo_completion_triplet.png and PLY files in outputs/.")


if __name__ == "__main__":
    main()
