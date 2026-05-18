from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import SyntheticCompletionDataset, save_ply, set_seed
from .models import ConditionalLatentDenoiser, DiffusionSchedule, PointNetEncoder, PointVAE, sample_conditional_ddpm
from .visualize import plot_triplet


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete partial point clouds using conditional latent diffusion.")
    parser.add_argument("--vae", type=str, default="checkpoints/point_vae.pt")
    parser.add_argument("--diffusion", type=str, default="checkpoints/point_cond_ddpm.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_ckpt = torch.load(args.vae, map_location=device)
    ddpm_ckpt = torch.load(args.diffusion, map_location=device)
    vae = PointVAE(vae_ckpt["latent_dim"], vae_ckpt["n_full"]).to(device)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()
    enc = PointNetEncoder(ddpm_ckpt["latent_dim"]).to(device)
    enc.load_state_dict(ddpm_ckpt["partial_encoder_state"])
    enc.eval()
    denoiser = ConditionalLatentDenoiser(ddpm_ckpt["latent_dim"], ddpm_ckpt["latent_dim"]).to(device)
    denoiser.load_state_dict(ddpm_ckpt["denoiser_state"])
    denoiser.eval()

    dataset = SyntheticCompletionDataset(num_shapes=max(8, args.index + 1), n_full=vae_ckpt["n_full"], n_partial=256, seed=args.seed)
    sample = dataset[args.index]
    partial = sample["partial"].unsqueeze(0).to(device)
    full = sample["full"].unsqueeze(0).to(device)
    with torch.no_grad():
        cond = enc(partial)
        schedule = DiffusionSchedule(timesteps=ddpm_ckpt["timesteps"])
        z = sample_conditional_ddpm(denoiser, cond, ddpm_ckpt["latent_dim"], schedule, device)
        pred = vae.decode(z)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_ply(partial.squeeze(0).cpu().numpy(), out_dir / "partial.ply")
    save_ply(pred.squeeze(0).cpu().numpy(), out_dir / "completed.ply")
    save_ply(full.squeeze(0).cpu().numpy(), out_dir / "ground_truth.ply")
    plot_triplet(partial.squeeze(0).cpu().numpy(), pred.squeeze(0).cpu().numpy(), full.squeeze(0).cpu().numpy(), out_dir / "completion_triplet.png")
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
