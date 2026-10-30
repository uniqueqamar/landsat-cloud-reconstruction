"""Evaluate fixed masks against a simple per-band observed-pixel mean baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import make_generator
from train import ClearPatches, TemporalPairs, choose_device, evaluate


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    device = choose_device()
    c = torch.load(a.weights, weights_only=True, map_location='cpu')
    dataset = (ClearPatches(a.data, a.split, seed=1000) if c['training'] == 'synthetic' else
               TemporalPairs(a.data, a.split))
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    model = make_generator(c).to(device)
    model.load_state_dict(c['model'])
    scores = evaluate(model, loader, device, c['training'], a.split)
    absolute = squared = count = 0.
    for image, mask, target in loader:
        observed_mean = (image * (1 - mask)).sum((2, 3), keepdim=True) / (1 - mask).sum((2, 3), keepdim=True).clamp_min(1)
        difference = observed_mean - target
        absolute += float((difference.abs() * mask).sum())
        squared += float((difference.square() * mask).sum())
        count += float(mask.sum()) * 4
    scores.update(mean_fill_masked_mae=absolute / count,
                  mean_fill_masked_psnr_unit_range_db=-10 * np.log10(max(squared / count, 1e-12)),
                  sample_count=len(dataset), training=c['training'],
                  note='Small single-scene experiment; temporal targets are not simultaneous ground truth.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
