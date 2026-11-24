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


# Fractions refer to QA-masked area, not to cloud optical thickness.
COVERAGE_LIMITS = (0.30, 0.50, 1.00)


@torch.no_grad()
def audit(model, loader, device, *, coverage_breakdown=False):
    model.eval()
    methods = ['model', 'mean_fill']
    if coverage_breakdown:
        from spatial_fill import spatial_fill

        methods += ['spatial_interpolation', 'cloudy_input']
    overall = {name: MaskedQuality() for name in methods}
    groups = [{name: MaskedQuality() for name in methods} for _ in COVERAGE_LIMITS]
    counts = [0] * len(COVERAGE_LIMITS)
    for observed, mask, target in loader:
        observed, mask, target = (v.to(device) for v in (observed, mask, target))
        valid = 1 - mask
        mean = (observed * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp_min(1)
        if torch.any(valid.sum((2, 3)) == 0):
            raise ValueError('Cannot compute observed-pixel mean fill: sample has no clear pixels')
        predictions = {'model': model(observed, mask), 'mean_fill': observed * valid + mean * mask}
        if coverage_breakdown:
            predictions['spatial_interpolation'] = spatial_fill(observed, mask)
            predictions['cloudy_input'] = observed
        masks, references = mask.cpu().numpy(), target.cpu().numpy()
        predictions = {name: value.cpu().numpy() for name, value in predictions.items()}
        for index, (patch_mask, reference) in enumerate(zip(masks, references)):
            if not patch_mask.any():
                continue
            if coverage_breakdown:
                fraction = int(patch_mask.sum()) / patch_mask.size
                group = next(i for i, limit in enumerate(COVERAGE_LIMITS) if fraction <= limit)
                counts[group] += 1
            for name, batches in predictions.items():
                overall[name].add(batches[index], reference, patch_mask)
                if coverage_breakdown:
                    groups[group][name].add(batches[index], reference, patch_mask)
    result = {name: quality.result() for name, quality in overall.items()}
    result['reference_caveat'] = 'Temporal reference is a later observation, not simultaneous ground truth.'
    if coverage_breakdown:
        result['coverage_groups'] = [
            {
                'masked_fraction_lower_exclusive': lower,
                'masked_fraction_upper_inclusive': upper,
                'patch_count': count,
                'metrics': {name: quality.result() for name, quality in scores.items()} if count else None,
            }
            for lower, upper, count, scores in zip((0.0, *COVERAGE_LIMITS[:-1]), COVERAGE_LIMITS, counts, groups)
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--coverage-breakdown', action='store_true',
                        help='Include cloudy input and spatial fill baselines, grouped by masked area')
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, weights_only=True, map_location='cpu')
    dataset = (ClearPatches(args.data, args.split, seed=1000)
               if checkpoint['training'] == 'synthetic' else TemporalPairs(args.data, args.split))
    model = make_generator(checkpoint)
    model.load_state_dict(checkpoint['model'])
    device = choose_device()
    model.to(device)
    scores = audit(model, DataLoader(dataset, batch_size=1, num_workers=0), device,
                   coverage_breakdown=args.coverage_breakdown)
    scores['split'] = args.split
    scores['sample_count'] = len(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
