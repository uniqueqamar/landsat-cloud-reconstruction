"""Train on clear patches with synthetic occlusions; no real paired targets assumed."""
import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import Discriminator, make_generator


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


CLOUD_TYPES = ('cloud', 'cirrus', 'shadow', 'dilated_only')


def cloud_type_ids(qa):
    """Assign each masked QA_PIXEL location to its highest-priority flag."""
    types = np.zeros(qa.shape, dtype=np.uint8)
    for index, bit in reversed(tuple(enumerate((3, 2, 4, 1), start=1))):
        types[(qa & (1 << bit)) != 0] = index
    return types


class TemporalPairs(Dataset):
    def __init__(self, root, split, expanded=False, augment=False,
                 shard_cache_limit_bytes=256 * 1024 * 1024, *, include_cloud_types=False):
        if shard_cache_limit_bytes < 0:
            raise ValueError('shard_cache_limit_bytes must be non-negative')
        self.root = root
        self.augment = augment
        self.include_cloud_types = include_cloud_types
        self.rng = np.random.default_rng(42)
        self.shard_cache = OrderedDict()
        self.shard_cache_bytes = 0
        self.shard_cache_limit_bytes = shard_cache_limit_bytes
        with (root / 'pairs.csv').open() as f:
            self.rows = [r for r in csv.DictReader(f) if r['split'] == split]
        if expanded:
            if split != 'train':
                raise ValueError('Expanded shards are training-only')
            with (root / 'expanded_train/index.csv').open() as f:
                self.rows.extend(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f'No temporal pairs for {split}')

    def __len__(self):
        return len(self.rows)

    def _get_shard(self, name):
        """Cache recently used expanded shards up to a bounded RAM budget.

        An oversized shard is decoded for the current sample but not retained.
        npz decoding still requires transient memory for one complete shard.
        """
        if name in self.shard_cache:
            self.shard_cache.move_to_end(name)
            return self.shard_cache[name]
        with np.load(self.root / name, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ['observed', 'target', 'qa_pixel']}
        size = sum(array.nbytes for array in arrays.values())
        if size <= self.shard_cache_limit_bytes:
            while self.shard_cache and self.shard_cache_bytes + size > self.shard_cache_limit_bytes:
                _, evicted = self.shard_cache.popitem(last=False)
                self.shard_cache_bytes -= sum(array.nbytes for array in evicted.values())
            self.shard_cache[name] = arrays
            self.shard_cache_bytes += size
        return arrays

    def __getitem__(self, i):
        row = self.rows[i]
        if 'shard' in row:
            z = self._get_shard(row['shard']); index = int(row['index'])
            observed = z['observed'][index].astype('float32') * .0000275 - .2
            reference = z['target'][index].astype('float32') * .0000275 - .2
            qa = z['qa_pixel'][index]
        else:
            with rasterio.open(self.root / row['input']) as src:
                observed = src.read([1, 2, 3, 4]).astype('float32') * .0000275 - .2
                qa = src.read(5)
            with rasterio.open(self.root / row['target']) as src:
                reference = src.read([1, 2, 3, 4]).astype('float32') * .0000275 - .2
        mask = ((qa & sum(1 << b for b in (1, 2, 3, 4))) != 0).astype('float32')[None]
        # Preserve observed pixels in the discriminator's real composite too.
        target = reference * mask + observed * (1 - mask)
        arrays = [observed, mask, target]
        if self.include_cloud_types:
            arrays.append(cloud_type_ids(qa)[None])
        if self.augment:
            rotations = int(self.rng.integers(0, 4))
            arrays = [np.rot90(x, rotations, axes=(-2, -1)).copy() for x in arrays]
            if self.rng.random() < .5:
                arrays = [x[..., ::-1].copy() for x in arrays]
        return tuple(torch.from_numpy(x) for x in arrays)


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def masked_mae(prediction, target, mask):
    return ((prediction - target).abs() * mask).sum() / (mask.sum() * 4).clamp_min(1)


def cloud_balanced_mae(prediction, target, cloud_types):
    """Average the class MAEs across the QA types present in the batch."""
    error = (prediction - target).abs().mean(dim=1, keepdim=True)
    present = cloud_types == torch.arange(
        1, len(CLOUD_TYPES) + 1, device=error.device, dtype=cloud_types.dtype
    ).view(1, -1, 1, 1)
    counts = present.sum(dim=(0, 2, 3))
    class_mae = (error * present).sum(dim=(0, 2, 3)) / counts.clamp_min(1)
    return class_mae.sum() / (counts > 0).sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, device, task, split='val'):
    model.eval()
    absolute = squared = original = count = 0.
    skipped_unmasked = 0
    for observed, mask, target in loader:
        observed, mask, target = [t.to(device) for t in (observed, mask, target)]
        # An entirely clear sample has no masked target pixels to score.
        skipped_unmasked += int((mask.flatten(1).sum(dim=1) == 0).sum())
        masked_pixels = float(mask.sum())
        if masked_pixels == 0:
            continue
        predicted = model(observed, mask)
        difference = predicted - target
        absolute += float((difference.abs() * mask).sum())
        squared += float((difference.square() * mask).sum())
        original += float(((observed - target).abs() * mask).sum())
        count += masked_pixels * 4
    if count == 0:
        raise ValueError(f'No masked pixels in {split} split; cannot compute masked metrics')
    return {'masked_mae': absolute / count,
            'masked_psnr_unit_range_db': -10 * np.log10(max(squared / count, 1e-12)),
            'occluded_input_masked_mae': original / count,
            'skipped_unmasked_samples': skipped_unmasked,
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
    p.add_argument('--architecture', choices=['legacy', 'context_residual_v2', 'srgan'], default='legacy')
    p.add_argument('--expanded-train', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--cloud-balance', action='store_true',
                   help='Blend pixel-weighted and QA-type-balanced reconstruction losses')
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--max-steps', type=int, default=0, help='Smoke test only; 0 means full run')
    a = p.parse_args()
    if a.cloud_balance and a.task != 'temporal':
        p.error('--cloud-balance requires real temporal QA_PIXEL flags')
    if a.architecture == 'srgan':
        if a.gan_weight <= 0 or not 0 <= a.warmup < a.epochs:
            p.error('SRGAN requires positive --gan-weight and at least one adversarial epoch')
        if a.max_steps and a.warmup:
            p.error('Use --warmup 0 for a step-limited SRGAN smoke test')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = choose_device()
    model = make_generator({'architecture': a.architecture, 'width': 16}).to(device)
    discriminator = None
    if a.gan_weight:
        discriminator = Discriminator(conditional=a.architecture != 'srgan').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4) if discriminator else None
    bce = nn.BCEWithLogitsLoss()
    train_data = ClearPatches(a.data, 'train') if a.task == 'synthetic' else TemporalPairs(
        a.data, 'train', expanded=a.expanded_train, augment=a.augment, include_cloud_types=a.cloud_balance)
    val_data = ClearPatches(a.data, 'val', seed=1000) if a.task == 'synthetic' else TemporalPairs(a.data, 'val')
    train = DataLoader(train_data, batch_size=a.batch_size,
                       shuffle=True, num_workers=0)
    val = DataLoader(val_data, batch_size=1, num_workers=0)
    print(f'Device: {device}; generator parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    history, best, steps = [], float('inf'), 0
    if a.architecture == 'context_residual_v2':
        # An unchanged mean fill is a valid incumbent; retain it if training fails.
        initial = evaluate(model, val, device, a.task)
        best = initial['masked_mae']
        history.append(dict(epoch=0, train_loss=None, **initial))
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        width=16, architecture=a.architecture, training=a.task,
                        bands=['B2','B3','B4','B5'], scale=.0000275, offset=-.2,
                        epoch=0, gan_weight=a.gan_weight, train_samples=len(train_data)), a.output/'best.pt')
        print(json.dumps(history[-1]), flush=True)
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for batch in train:
            observed, mask, target, *cloud_types = [t.to(device) for t in batch]
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
            if a.cloud_balance:
                loss = 0.5 * (loss + cloud_balanced_mae(predicted, target, cloud_types[0]))
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
                          gan_weight=a.gan_weight, epoch=epoch + 1,
                          architecture=a.architecture, train_samples=len(train_data),
                          expanded_train=a.expanded_train, augment=a.augment,
                          learning_rate=a.learning_rate, cloud_balance=a.cloud_balance)
        torch.save(checkpoint, a.output / 'last.pt')
        eligible = a.architecture != 'srgan' or epoch >= a.warmup
        if eligible and scores['masked_mae'] < best:
            best = scores['masked_mae']
            torch.save(checkpoint, a.output / 'best.pt')
        (a.output / 'history.json').write_text(json.dumps(history, indent=2))
        if a.max_steps and steps >= a.max_steps:
            break


if __name__ == '__main__':
    main()
