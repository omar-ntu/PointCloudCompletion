# Diffusion Model for 3D Point Cloud Completion

<i>Can a diffusion model recover missing 3D geometry from partial point-cloud observations?</i>

<img width="756" height="250" alt="sample_completion_triplet" src="https://github.com/user-attachments/assets/f8d09e41-8d98-47b3-8287-979b303928cb" />
<img width="756" height="250" alt="demo_completion_triplet" src="https://github.com/user-attachments/assets/6202ad2f-7cf0-4180-b78c-50ad08f02d85" />


This project implements a compact research prototype for completing partial 3D point clouds using three progressively stronger approaches:

1. **Autoencoder baseline**: encodes a partial point cloud and decodes a complete point cloud.
2. **VAE reconstruction model**: learns a latent distribution for full shapes.
3. **Conditional latent diffusion model**: denoises full-shape latents conditioned on partial point-cloud observations.

The default dataset is synthetic and generated from analytical 3D shapes: spheres, cubes, and cylinders. This makes the project runnable without downloading ShapeNet, ModelNet40, ScanNet, or other large datasets. The code structure is designed so the dataset can be replaced with real partial/full point-cloud pairs later.

## Repository structure

```text
point_cloud_completion_diffusion/
├── README.md
├── requirements.txt
├── run_demo.py
├── src/
│   ├── data.py                         # Synthetic full/partial point-cloud dataset
│   ├── losses.py                       # Chamfer Distance and F-score
│   ├── models.py                       # PointNet, AE, VAE, conditional latent DDPM
│   ├── train_autoencoder.py            # Baseline AE training
│   ├── train_vae.py                    # Full-shape VAE training
│   ├── train_diffusion_completion.py   # Conditional latent diffusion training
│   ├── complete_point_cloud.py         # Completion inference script
│   ├── evaluate.py                     # AE/VAE/DDPM evaluation
│   └── visualize.py                    # 3D scatter visualisation
└── outputs/
    ├── sample_partial.ply
    ├── sample_ground_truth.ply
    └── sample_completion_triplet.png
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick demo

```bash
python run_demo.py
```

This trains very small models briefly and writes:

```text
outputs/demo_partial.ply
outputs/demo_completed.ply
outputs/demo_ground_truth.ply
outputs/demo_completion_triplet.png
```

The demo is only a smoke test. For meaningful results, use the full training pipeline below.

## Full training pipeline

Train the autoencoder baseline:

```bash
python -m src.train_autoencoder \
  --num-shapes 1200 \
  --n-full 512 \
  --n-partial 256 \
  --epochs 30 \
  --out checkpoints/point_ae.pt
```

Train the VAE on complete point clouds:

```bash
python -m src.train_vae \
  --num-shapes 1200 \
  --n-full 512 \
  --n-partial 256 \
  --epochs 40 \
  --out checkpoints/point_vae.pt
```

Train conditional latent diffusion:

```bash
python -m src.train_diffusion_completion \
  --vae checkpoints/point_vae.pt \
  --num-shapes 1200 \
  --n-full 512 \
  --n-partial 256 \
  --epochs 40 \
  --out checkpoints/point_cond_ddpm.pt
```

Run completion inference:

```bash
python -m src.complete_point_cloud \
  --vae checkpoints/point_vae.pt \
  --diffusion checkpoints/point_cond_ddpm.pt \
  --index 0 \
  --out-dir outputs
```

Evaluate available checkpoints:

```bash
python -m src.evaluate
```

## Approach

The project uses a latent diffusion setup rather than directly diffusing thousands of 3D point coordinates.

First, the VAE learns a compact latent representation of complete point clouds. Then the conditional diffusion model is trained to generate a clean full-shape latent from random noise while conditioning on a partial point-cloud embedding. The completed point cloud is obtained by decoding the generated latent using the VAE decoder.

This gives the project three useful comparisons:

- **Autoencoder baseline**: deterministic partial-to-complete reconstruction.
- **VAE reconstruction**: generative latent representation of full shapes.
- **Conditional diffusion completion**: stochastic completion conditioned on visible geometry.

## Metrics

The included evaluator reports:

- **Chamfer Distance**: average nearest-neighbour distance between predicted and ground-truth point clouds. Lower is better.
- **F-score@0.05**: percentage-style geometric overlap under a distance threshold. Higher is better.

For real datasets, you may also add Earth Mover's Distance, normal consistency, category-wise metrics, and visual inspection.

## Data format

The synthetic dataset returns:

```python
{
    "partial": Tensor[n_partial, 3],
    "full": Tensor[n_full, 3],
    "label": Tensor[]
}
```

All point clouds are centred and normalised to fit approximately inside a unit sphere.
