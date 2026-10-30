"""Train on clear patches with synthetic occlusions; no real paired targets assumed."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import Generator, Discriminator


def synthetic_mask(height, width, rng):
    yy, xx = np.mgrid[:height, :width]
    mask = np.zeros((height, width), bool)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(.15, .85, 2) * [width, height]
        rx, ry = rng.uniform(.12, .30, 2) * [width, height]
        mask |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    return mask.astype('float32')[None]


class ClearPatches(Dataset):
    def __init__(self, root, split, seed=None):
        self.paths = sorted((root / 'data' / 'clear' / split).glob('*.tif'))
        self.seed = seed
        self.rng = np.random.default_rng(42)
        if not self.paths:
            raise ValueError(f'No clear patches for {split}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with rasterio.open(self.paths[i]) as src:
            target = src.read([1, 2, 3, 4]).astype('float32') * .0000275 - .2
        rng = self.rng if self.seed is None else np.random.default_rng(self.seed + i)
        mask = synthetic_mask(*target.shape[-2:], rng)
        # Synthetic opaque occlusion, not a physical cloud simulator.
        color = rng.uniform(.65, 1.0, (4, 1, 1)).astype('float32')
        observed = target * (1 - mask) + color * mask
        return tuple(torch.from_numpy(x) for x in (observed, mask, target))


class TemporalPairs(Dataset):
    def __init__(self, root, split):
        self.root = root
        with (root / 'pairs.csv').open() as f:
            self.rows = [r for r in csv.DictReader(f) if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No temporal pairs for {split}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with rasterio.open(self.root / row['input']) as src:
            observed = src.read([1, 2, 3, 4]).astype('float32') * .0000275 - .2
            qa = src.read(5)
        with rasterio.open(self.root / row['target']) as src:
            reference = src.read([1, 2, 3, 4]).astype('float32') * .0000275 - .2
        mask = ((qa & sum(1 << b for b in (1, 2, 3, 4))) != 0).astype('float32')[None]
        # Preserve observed pixels in the discriminator's real composite too.
        target = reference * mask + observed * (1 - mask)
        return tuple(torch.from_numpy(x) for x in (observed, mask, target))


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def masked_mae(prediction, target, mask):
    return ((prediction - target).abs() * mask).sum() / (mask.sum() * 4).clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, device, task, split='val'):
    model.eval()
    absolute = squared = original = count = 0.
    for observed, mask, target in loader:
        observed, mask, target = [t.to(device) for t in (observed, mask, target)]
        predicted = model(observed, mask)
        difference = predicted - target
        absolute += float((difference.abs() * mask).sum())
        squared += float((difference.square() * mask).sum())
        original += float(((observed - target).abs() * mask).sum())
        count += float(mask.sum()) * 4
    return {'masked_mae': absolute / count,
            'masked_psnr_unit_range_db': -10 * np.log10(max(squared / count, 1e-12)),
            'occluded_input_masked_mae': original / count,
            'evaluation': ('fixed synthetic occlusions' if task == 'synthetic' else
                           'real cloudy inputs with eight-day temporal proxy targets') + f', {split} split'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path, default=Path('runs/baseline'))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gan-weight', type=float, default=0.)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--task', choices=['synthetic', 'temporal'], default='synthetic')
    p.add_argument('--max-steps', type=int, default=0, help='Smoke test only; 0 means full run')
    a = p.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = choose_device()
    model = Generator().to(device)
    discriminator = Discriminator().to(device) if a.gan_weight else None
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4) if discriminator else None
    bce = nn.BCEWithLogitsLoss()
    train_data = ClearPatches(a.data, 'train') if a.task == 'synthetic' else TemporalPairs(a.data, 'train')
    val_data = ClearPatches(a.data, 'val', seed=1000) if a.task == 'synthetic' else TemporalPairs(a.data, 'val')
    train = DataLoader(train_data, batch_size=a.batch_size,
                       shuffle=True, num_workers=0)
    val = DataLoader(val_data, batch_size=1, num_workers=0)
    print(f'Device: {device}; generator parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    history, best, steps = [], float('inf'), 0
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for observed, mask, target in train:
            observed, mask, target = [t.to(device) for t in (observed, mask, target)]
            predicted = model(observed, mask)
            adversarial = discriminator is not None and epoch >= a.warmup
            if adversarial:
                for t in discriminator.parameters():
                    t.requires_grad_(True)
                d_optimizer.zero_grad(set_to_none=True)
                real, fake = discriminator(observed, mask, target), discriminator(observed, mask, predicted.detach())
                d_loss = (bce(real, torch.ones_like(real)) + bce(fake, torch.zeros_like(fake))) / 2
                d_loss.backward()
                d_optimizer.step()
                for t in discriminator.parameters():
                    t.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mae(predicted, target, mask)
            if adversarial:
                logits = discriminator(observed, mask, predicted)
                loss = loss + a.gan_weight * bce(logits, torch.ones_like(logits))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
            if a.max_steps and steps >= a.max_steps:
                break
        scores = evaluate(model, val, device, a.task)
        record = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), **scores)
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, width=16,
                          bands=['B2', 'B3', 'B4', 'B5'], scale=.0000275, offset=-.2,
                          training=a.task, temporal_proxy_targets=a.task == 'temporal',
                          gan_weight=a.gan_weight, epoch=epoch + 1)
        torch.save(checkpoint, a.output / 'last.pt')
        if scores['masked_mae'] < best:
            best = scores['masked_mae']
            torch.save(checkpoint, a.output / 'best.pt')
        (a.output / 'history.json').write_text(json.dumps(history, indent=2))
        if a.max_steps and steps >= a.max_steps:
            break


if __name__ == '__main__':
    main()
