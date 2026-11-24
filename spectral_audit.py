"""Compare reconstructed reflectance with a reference inside the cloud mask."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import make_generator
from quality_metrics import MaskedQuality
from spatial_fill import spatial_fill
from train import ClearPatches, TemporalPairs, choose_device


# Bounds describe the QA-masked fraction of a patch.
COVERAGE_RANGES = ((0.0, 0.30), (0.30, 0.50), (0.50, 1.00))


@torch.no_grad()
def audit(model, loader, device, *, coverage_breakdown=False):
    """Pool error across masked pixels and optionally group patches by coverage."""
    model.eval()
    methods = ['model', 'mean_fill']
    if coverage_breakdown:
        methods += ['spatial_interpolation', 'cloudy_input']
    overall_scores = {name: MaskedQuality() for name in methods}
    coverage_scores = [{name: MaskedQuality() for name in methods} for _ in COVERAGE_RANGES]
    patch_counts = [0] * len(COVERAGE_RANGES)
    for observed, mask, target in loader:
        observed, mask, target = (tensor.to(device) for tensor in (observed, mask, target))
        clear = 1 - mask
        clear_count = clear.sum((2, 3), keepdim=True)
        if torch.any(clear_count == 0):
            raise ValueError('Cannot compute observed-pixel mean fill: sample has no clear pixels')
        clear_mean = (observed * clear).sum((2, 3), keepdim=True) / clear_count.clamp_min(1)
        predictions = {'model': model(observed, mask), 'mean_fill': observed * clear + clear_mean * mask}
        if coverage_breakdown:
            predictions['spatial_interpolation'] = spatial_fill(observed, mask)
            predictions['cloudy_input'] = observed
        masks = mask.detach().cpu().numpy()
        references = target.detach().cpu().numpy()
        predictions = {name: value.detach().cpu().numpy() for name, value in predictions.items()}
        for index, (patch_mask, reference) in enumerate(zip(masks, references)):
            if not patch_mask.any():
                continue
            if coverage_breakdown:
                masked_fraction = int(patch_mask.sum()) / patch_mask.size
                group_index = next(i for i, (_, upper) in enumerate(COVERAGE_RANGES)
                                   if masked_fraction <= upper)
                patch_counts[group_index] += 1
            for name, batch in predictions.items():
                overall_scores[name].add(batch[index], reference, patch_mask)
                if coverage_breakdown:
                    coverage_scores[group_index][name].add(batch[index], reference, patch_mask)
    result = {name: quality.result() for name, quality in overall_scores.items()}
    result['reference_caveat'] = 'Temporal reference is a later observation, not simultaneous ground truth.'
    if coverage_breakdown:
        result['coverage_groups'] = [
            {
                'masked_fraction_lower_exclusive': lower,
                'masked_fraction_upper_inclusive': upper,
                'patch_count': count,
                'metrics': {name: quality.result() for name, quality in scores.items()} if count else None,
            }
            for (lower, upper), count, scores in zip(COVERAGE_RANGES, patch_counts, coverage_scores)
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
