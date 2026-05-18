"""Synthetic point-cloud completion dataset.

The default dataset samples clean analytical shapes, creates partial observations by
simulating a single-view crop, and provides full point clouds as reconstruction
targets. This keeps the project runnable without ShapeNet/ModelNet downloads.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

SHAPE_CLASSES = ["sphere", "cube", "cylinder"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32)
    ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]], dtype=np.float32)
    return rz @ ry


def normalize(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pc, axis=1)) + 1e-8
    return (pc / scale).astype(np.float32)


def sample_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(0, 1, size=n)
    v = rng.uniform(0, 1, size=n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pc = np.stack([x, y, z], axis=1)
    pc += rng.normal(0, 0.01, pc.shape)
    return normalize(pc)


def sample_cube(n: int, rng: np.random.Generator) -> np.ndarray:
    pts = []
    for _ in range(n):
        face = rng.integers(0, 6)
        p = rng.uniform(-1, 1, size=3)
        axis = face // 2
        sign = 1 if face % 2 == 0 else -1
        p[axis] = sign
        pts.append(p)
    pc = np.asarray(pts, dtype=np.float32)
    pc += rng.normal(0, 0.01, pc.shape)
    return normalize(pc @ random_rotation(rng).T)


def sample_cylinder(n: int, rng: np.random.Generator) -> np.ndarray:
    pts = []
    for _ in range(n):
        part = rng.choice(["side", "top", "bottom"], p=[0.65, 0.175, 0.175])
        theta = rng.uniform(0, 2 * np.pi)
        if part == "side":
            r = 1.0
            z = rng.uniform(-1, 1)
        else:
            r = math.sqrt(rng.uniform(0, 1))
            z = 1.0 if part == "top" else -1.0
        pts.append([r * math.cos(theta), r * math.sin(theta), z])
    pc = np.asarray(pts, dtype=np.float32)
    pc += rng.normal(0, 0.01, pc.shape)
    return normalize(pc @ random_rotation(rng).T)


def sample_shape(shape: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if shape == "sphere":
        return sample_sphere(n, rng)
    if shape == "cube":
        return sample_cube(n, rng)
    if shape == "cylinder":
        return sample_cylinder(n, rng)
    raise ValueError(f"Unknown shape: {shape}")


def make_partial(full_pc: np.ndarray, n_partial: int, rng: np.random.Generator) -> np.ndarray:
    """Keep points most visible from a random viewing direction, then resample."""
    view = rng.normal(size=3)
    view = view / (np.linalg.norm(view) + 1e-8)
    scores = full_pc @ view
    threshold = np.quantile(scores, 0.45)
    visible = full_pc[scores >= threshold]
    if len(visible) == 0:
        visible = full_pc
    idx = rng.choice(len(visible), size=n_partial, replace=len(visible) < n_partial)
    partial = visible[idx]
    partial += rng.normal(0, 0.005, partial.shape)
    return partial.astype(np.float32)


class SyntheticCompletionDataset(Dataset):
    def __init__(self, num_shapes: int = 1200, n_full: int = 512, n_partial: int = 256, seed: int = 42):
        self.num_shapes = num_shapes
        self.n_full = n_full
        self.n_partial = n_partial
        self.seed = seed
        rng = np.random.default_rng(seed)
        partials, fulls, labels = [], [], []
        for i in range(num_shapes):
            label = i % len(SHAPE_CLASSES)
            shape = SHAPE_CLASSES[label]
            full = sample_shape(shape, n_full, rng)
            partial = make_partial(full, n_partial, rng)
            partials.append(partial)
            fulls.append(full)
            labels.append(label)
        self.partials = np.stack(partials).astype(np.float32)
        self.fulls = np.stack(fulls).astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_shapes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "partial": torch.from_numpy(self.partials[idx]),
            "full": torch.from_numpy(self.fulls[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def save_ply(points: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ])
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
