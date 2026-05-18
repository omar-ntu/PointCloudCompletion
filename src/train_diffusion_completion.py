from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SyntheticCompletionDataset, set_seed
from .models import ConditionalLatentDenoiser, DiffusionSchedule, PointNetEncoder, PointVAE, q_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional latent diffusion for point-cloud completion.")
    parser.add_argument("--vae", type=str, default="checkpoints/point_vae.pt")
    parser.add_argument("--num-shapes", type=int, default=1200)
    parser.add_argument("--n-full", type=int, default=512)
    parser.add_argument("--n-partial", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoints/point_cond_ddpm.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_ckpt = torch.load(args.vae, map_location=device)
    vae = PointVAE(vae_ckpt["latent_dim"], vae_ckpt["n_full"]).to(device)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    partial_encoder = PointNetEncoder(args.latent_dim).to(device)
    denoiser = ConditionalLatentDenoiser(latent_dim=args.latent_dim, cond_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(list(partial_encoder.parameters()) + list(denoiser.parameters()), lr=args.lr, weight_decay=1e-4)
    schedule = DiffusionSchedule(timesteps=args.timesteps)
    _, _, alpha_bars = schedule.tensors(device)

    dataset = SyntheticCompletionDataset(args.num_shapes, args.n_full, args.n_partial, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch:03d}"):
            partial = batch["partial"].to(device)
            full = batch["full"].to(device)
            with torch.no_grad():
                z0, _ = vae.encode(full)
            cond = partial_encoder(partial)
            t = torch.randint(0, args.timesteps, (partial.size(0),), device=device)
            noise = torch.randn_like(z0)
            zt = q_sample(z0, t, noise, alpha_bars)
            pred_noise = denoiser(zt, cond, t)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch:03d} diffusion_loss={sum(losses)/len(losses):.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "denoiser_state": denoiser.state_dict(),
        "partial_encoder_state": partial_encoder.state_dict(),
        "latent_dim": args.latent_dim,
        "n_full": args.n_full,
        "timesteps": args.timesteps,
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
