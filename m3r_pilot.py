"""Train or run the matched optical and SAR models on the public M3R-CR examples."""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch
from torch import nn

from evaluate_m3r import read_optical


class Reconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(8, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, visible_mean):
        return (visible_mean + 0.1 * self.layers(features)).clamp(0, 1)


def radar_on_grid(path, reference_grid):
    crs, transform, height, width = reference_grid
    with rasterio.open(path) as source:
        if source.crs is None or source.count != 2:
            raise ValueError(f"{path}: expected a georeferenced two-band SAR TIFF")
        raw = source.read(masked=True, out_dtype="float32").filled(np.nan)
        floor = np.array([-25, -32.5], dtype=np.float32)[:, None, None]
        normalized = np.clip((raw - floor) / -floor, 0, 1)
        projected = np.full((2, height, width), np.nan, dtype=np.float32)
        for channel in range(2):
            reproject(
                normalized[channel], projected[channel],
                src_crs=source.crs, src_transform=source.transform, src_nodata=np.nan,
                dst_crs=crs, dst_transform=transform, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    valid = np.isfinite(projected).all(axis=0)
    if not valid.any():
        raise ValueError(f"{path}: SAR has no valid overlap with the optical patch")
    projected[:, ~valid] = 0.5
    return projected - 0.5, valid[None]


def load_example(root, split):
    with (root / f"one_{split}_sample.csv").open(newline="") as source:
        entries = {tuple(row) for row in csv.reader(source) if row}
    if len(entries) != 1:
        raise ValueError(f"Expected one unique upstream {split} example")
    reference_path, cloudy_path, sar_path, _, mask_path = [
        root / split / name for name in entries.pop()
    ]
    with rasterio.open(reference_path) as template:
        profile = template.profile
        reference_grid = (template.crs, template.transform, template.height, template.width)
    observed, _ = read_optical(cloudy_path, reference_grid)
    observed = np.clip(observed, 0, 10000, out=observed, where=np.isfinite(observed)) / 10000
    mask = np.load(mask_path, allow_pickle=False)
    if mask.shape != observed.shape[1:] or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{mask_path}: expected a binary mask matching the optical grid")
    mask = mask.astype(bool)
    visible = ~mask & np.isfinite(observed).all(axis=0)
    if not visible.any() or not mask.any():
        raise ValueError(f"{cloudy_path}: expected a partially cloudy patch")
    mean = observed[:, visible].mean(axis=1)[:, None, None]
    centered = np.where(visible, (observed - mean) / 0.1, 0)
    optical = np.concatenate((centered, ~visible[None]), axis=0)
    radar, valid = radar_on_grid(sar_path, reference_grid)
    guided = np.concatenate((optical, radar, valid), axis=0)
    optical = np.concatenate((optical, np.zeros_like(guided[5:])), axis=0)
    features = {
        "optical_only": torch.from_numpy(optical[None]),
        "sar_guided": torch.from_numpy(guided[None]),
    }
    return reference_path, profile, features, torch.from_numpy(mean[None]), observed, mask


def train(root, checkpoint_dir):
    reference_path, _, features, mean, _, mask = load_example(root, "train")
    target, _ = read_optical(reference_path)
    valid = mask & np.isfinite(target).all(axis=0)
    if not valid.any():
        raise ValueError("Training example has no valid reference under the cloud mask")
    target = torch.from_numpy(np.clip(target[:, valid], 0, 10000) / 10000)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for method, inputs in features.items():
        torch.manual_seed(42)
        model = Reconstruction()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(100):
            optimizer.zero_grad()
            error = nn.functional.l1_loss(model(inputs, mean)[0, :, valid], target)
            error.backward()
            optimizer.step()
        torch.save({
            "model": model.state_dict(), "method": method,
            "train_patch": reference_path.stem, "steps": 100, "seed": 42,
        }, checkpoint_dir / f"{method}.pt")
        print(f"{method} final training step masked MAE: {error.item():.6f}")


@torch.inference_mode()
def predict(root, checkpoint_dir, output):
    reference_path, profile, features, mean, observed, mask = load_example(root, "test")
    profile.update(dtype="float32", nodata=np.nan)
    for method, inputs in features.items():
        checkpoint = torch.load(checkpoint_dir / f"{method}.pt", map_location="cpu", weights_only=True)
        if checkpoint["method"] != method or checkpoint["train_patch"] == reference_path.stem:
            raise ValueError(f"Unexpected checkpoint metadata for {method}")
        model = Reconstruction().eval()
        model.load_state_dict(checkpoint["model"])
        prediction = np.where(mask, model(inputs, mean)[0].numpy(), observed)
        directory = output / method
        directory.mkdir(parents=True, exist_ok=True)
        with rasterio.open(directory / reference_path.name, "w", **profile) as destination:
            destination.write(prediction)
            destination.descriptions = ("blue", "green", "red", "near infrared")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "predict"))
    parser.add_argument("--dataset", required=True, type=Path, help="Upstream M3M-CR directory")
    parser.add_argument("--weights", type=Path, default=Path("runs/m3r_public_pilot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/m3r_pilot"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "train":
        train(args.dataset, args.weights)
    else:
        predict(args.dataset, args.weights, args.output)


if __name__ == "__main__":
    main()
