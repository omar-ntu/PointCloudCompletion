from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SyntheticCompletionDataset, set_seed
from .losses import chamfer_distance
from .models import PointAutoEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PointNet autoencoder baseline for point-cloud completion.")
    parser.add_argument("--num-shapes", type=int, default=1200)
    parser.add_argument("--n-full", type=int, default=512)
    parser.add_argument("--n-partial", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="checkpoints/point_ae.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticCompletionDataset(args.num_shapes, args.n_full, args.n_partial, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = PointAutoEncoder(args.latent_dim, args.n_full).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch:03d}"):
            partial, full = batch["partial"].to(device), batch["full"].to(device)
            pred = model(partial)
            loss = chamfer_distance(pred, full)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch:03d} chamfer={sum(losses)/len(losses):.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "latent_dim": args.latent_dim, "n_full": args.n_full}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
