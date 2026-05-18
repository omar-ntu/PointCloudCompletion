from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SyntheticCompletionDataset
from .losses import chamfer_distance, fscore
from .models import (
    ConditionalLatentDenoiser,
    DiffusionSchedule,
    PointAutoEncoder,
    PointNetEncoder,
    PointVAE,
    sample_conditional_ddpm,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AE, VAE, and diffusion completion models if checkpoints exist.")
    parser.add_argument("--num-shapes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-full", type=int, default=512)
    parser.add_argument("--n-partial", type=int, default=256)
    parser.add_argument("--ae", type=str, default="checkpoints/point_ae.pt")
    parser.add_argument("--vae", type=str, default="checkpoints/point_vae.pt")
    parser.add_argument("--diffusion", type=str, default="checkpoints/point_cond_ddpm.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticCompletionDataset(args.num_shapes, args.n_full, args.n_partial, seed=999)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    results = {}

    if Path(args.ae).exists():
        ckpt = torch.load(args.ae, map_location=device)
        model = PointAutoEncoder(ckpt["latent_dim"], ckpt["n_full"]).to(device)
        model.load_state_dict(ckpt["model_state"]); model.eval()
        cds, fs = [], []
        for batch in loader:
            partial, full = batch["partial"].to(device), batch["full"].to(device)
            pred = model(partial)
            cds.append(float(chamfer_distance(pred, full).cpu()))
            fs.append(float(fscore(pred, full).cpu()))
        results["autoencoder"] = {"chamfer": sum(cds) / len(cds), "fscore@0.05": sum(fs) / len(fs)}

    if Path(args.vae).exists():
        ckpt = torch.load(args.vae, map_location=device)
        model = PointVAE(ckpt["latent_dim"], ckpt["n_full"]).to(device)
        model.load_state_dict(ckpt["model_state"]); model.eval()
        cds, fs = [], []
        for batch in loader:
            full = batch["full"].to(device)
            recon, _, _, _ = model(full)
            cds.append(float(chamfer_distance(recon, full).cpu()))
            fs.append(float(fscore(recon, full).cpu()))
        results["vae_reconstruction"] = {"chamfer": sum(cds) / len(cds), "fscore@0.05": sum(fs) / len(fs)}

    if Path(args.vae).exists() and Path(args.diffusion).exists():
        vae_ckpt = torch.load(args.vae, map_location=device)
        ddpm_ckpt = torch.load(args.diffusion, map_location=device)
        vae = PointVAE(vae_ckpt["latent_dim"], vae_ckpt["n_full"]).to(device)
        vae.load_state_dict(vae_ckpt["model_state"]); vae.eval()
        enc = PointNetEncoder(ddpm_ckpt["latent_dim"]).to(device)
        enc.load_state_dict(ddpm_ckpt["partial_encoder_state"]); enc.eval()
        denoiser = ConditionalLatentDenoiser(ddpm_ckpt["latent_dim"], ddpm_ckpt["latent_dim"]).to(device)
        denoiser.load_state_dict(ddpm_ckpt["denoiser_state"]); denoiser.eval()
        schedule = DiffusionSchedule(timesteps=ddpm_ckpt["timesteps"])
        cds, fs = [], []
        for batch in loader:
            partial, full = batch["partial"].to(device), batch["full"].to(device)
            cond = enc(partial)
            z = sample_conditional_ddpm(denoiser, cond, ddpm_ckpt["latent_dim"], schedule, device)
            pred = vae.decode(z)
            cds.append(float(chamfer_distance(pred, full).cpu()))
            fs.append(float(fscore(pred, full).cpu()))
        results["conditional_latent_diffusion"] = {"chamfer": sum(cds) / len(cds), "fscore@0.05": sum(fs) / len(fs)}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
