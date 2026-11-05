"""Independent spectral audit; use validation only during model development.

This is a diagnostic, not evidence of geographic generalization.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import make_generator
from train import ClearPatches, TemporalPairs, choose_device
from quality_metrics import MaskedQuality


@torch.no_grad()
def audit(model, loader, device):
    model.eval()
    model_quality, mean_quality = MaskedQuality(), MaskedQuality()
    for observed, mask, target in loader:
        observed, mask, target = (v.to(device) for v in (observed, mask, target))
        prediction = model(observed, mask)
        valid = 1 - mask
        mean = (observed * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp_min(1)
        baseline = observed * valid + mean * mask
        for j in range(observed.shape[0]):
            m = mask[j].detach().cpu().numpy()
            t = target[j].detach().cpu().numpy()
            model_quality.add(prediction[j].detach().cpu().numpy(), t, m)
            mean_quality.add(baseline[j].detach().cpu().numpy(), t, m)
    return {'model': model_quality.result(), 'mean_fill': mean_quality.result(),
            'reference_caveat': 'Temporal reference is a later observation, not simultaneous ground truth.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, weights_only=True, map_location='cpu')
    dataset = (ClearPatches(args.data, args.split, seed=1000)
               if checkpoint['training'] == 'synthetic' else TemporalPairs(args.data, args.split))
    model = make_generator(checkpoint)
    model.load_state_dict(checkpoint['model'])
    device = choose_device()
    model.to(device)
    scores = audit(model, DataLoader(dataset, batch_size=1, num_workers=0), device)
    scores['split'] = args.split
    scores['sample_count'] = len(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
