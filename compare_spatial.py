"""Compare spatial interpolation with mean filling on held-out patches.

Later temporal targets are proxies, not cloud-hidden ground truth.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import ClearPatches, TemporalPairs
from spatial_fill import spatial_fill


@torch.no_grad()
def compare(loader, iterations=128):
    totals = {'spatial_absolute': 0., 'mean_absolute': 0., 'pixels': 0., 'samples': 0}
    for observed, mask, target in loader:
        valid = 1 - mask
        if torch.any(valid.sum((2, 3)) == 0):
            raise ValueError('No observed pixels: spatial and mean baselines undefined')
        if mask.sum() == 0:
            continue
        mean = (observed * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True)
        prediction = spatial_fill(observed, mask, iterations)
        totals['spatial_absolute'] += ((prediction - target).abs() * mask).sum().item()
        totals['mean_absolute'] += ((mean - target).abs() * mask).sum().item()
        totals['pixels'] += mask.sum().item() * 4
        totals['samples'] += observed.shape[0]
    if not totals['pixels']:
        raise ValueError('No masked pixels in evaluation split')
    return {'spatial_masked_mae': totals['spatial_absolute'] / totals['pixels'],
            'mean_fill_masked_mae': totals['mean_absolute'] / totals['pixels'],
            'evaluated_samples': totals['samples'], 'masked_band_pixels': int(totals['pixels']),
            'iterations': iterations,
            'reference_caveat': 'Later optical proxy; not true cloud-hidden ground truth'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--task', choices=['temporal', 'synthetic'], default='temporal')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--iterations', type=int, default=128)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    dataset = TemporalPairs(a.data, a.split) if a.task == 'temporal' else ClearPatches(a.data, a.split, seed=1000)
    result = compare(DataLoader(dataset, batch_size=1, num_workers=0), a.iterations)
    result.update(task=a.task, split=a.split)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
