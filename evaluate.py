"""Evaluate masked reconstruction and observed-pixel mean fill in one data pass.

All comparisons use the same fixed masks and targets. Temporal references are
later acquisitions, not ground truth for details hidden by the original cloud.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_with_baseline(model, loader, device, training, split='val', sample_ids=None):
    """Compute pixel-weighted metrics plus transparent per-patch paired errors.

    The loader uses batch_size=1 to keep RAM usage low and retain sample IDs.
    A patch with no masked or no clear pixels is rejected rather than silently
    returning a misleading score or a zero-valued mean-fill baseline.
    """
    model.eval()
    model_abs = model_sq = original_abs = baseline_abs = baseline_sq = count = 0.0
    per_patch = []
    for index, (image, mask, target) in enumerate(loader):
        if image.ndim != 4 or image.shape[0] != 1 or mask.shape != image.shape[:1] + (1,) + image.shape[2:]:
            raise ValueError('Evaluation requires batch_size=1 and a [N,1,H,W] mask')
        image, mask, target = [t.to(device) for t in (image, mask, target)]
        masked_pixels = float(mask.sum().item())
        clear_pixels = float((1 - mask).sum().item())
        if masked_pixels <= 0 or clear_pixels <= 0:
            raise ValueError(f'Sample {index} must contain both masked and clear pixels')
        channels = target.shape[1]
        denominator = masked_pixels * channels
        observed_mean = (image * (1 - mask)).sum((2, 3), keepdim=True) / (1 - mask).sum((2, 3), keepdim=True)
        predicted = model(image, mask)
        model_error = (predicted - target) * mask
        baseline_error = (observed_mean - target) * mask
        original_error = (image - target) * mask
        sample_model_abs = float(model_error.abs().sum().item())
        sample_baseline_abs = float(baseline_error.abs().sum().item())
        model_abs += sample_model_abs
        model_sq += float(model_error.square().sum().item())
        original_abs += float(original_error.abs().sum().item())
        baseline_abs += sample_baseline_abs
        baseline_sq += float(baseline_error.square().sum().item())
        count += denominator
        model_mae = sample_model_abs / denominator
        baseline_mae = sample_baseline_abs / denominator
        per_patch.append({
            'sample': sample_ids[index] if sample_ids is not None else str(index),
            'masked_fraction': masked_pixels / mask[0, 0].numel(),
            'model_masked_mae': model_mae,
            'mean_fill_masked_mae': baseline_mae,
            'model_minus_mean_fill_mae': model_mae - baseline_mae,
        })
    if not per_patch:
        raise ValueError('Cannot evaluate an empty dataset')
    differences = np.array([r['model_minus_mean_fill_mae'] for r in per_patch])
    return {
        'masked_mae': model_abs / count,
        'masked_psnr_unit_range_db': float(-10 * np.log10(max(model_sq / count, 1e-12))),
        'occluded_input_masked_mae': original_abs / count,
        'evaluation': ('fixed synthetic occlusions' if training == 'synthetic' else
                       'real cloudy inputs with eight-day temporal proxy targets') + f', {split} split',
        'mean_fill_masked_mae': baseline_abs / count,
        'mean_fill_masked_psnr_unit_range_db': float(-10 * np.log10(max(baseline_sq / count, 1e-12))),
        'sample_count': len(per_patch),
        'training': training,
        'model_better_patch_count': int((differences < -1e-10).sum()),
        'mean_fill_better_patch_count': int((differences > 1e-10).sum()),
        'tied_patch_count': int((np.abs(differences) <= 1e-10).sum()),
        'median_patch_mae_delta': float(np.median(differences)),
        'per_patch': per_patch,
        'note': 'Small single-scene experiment; temporal targets are not simultaneous ground truth. '
                'Overall MAE is weighted by masked pixels; patch counts and median are unweighted.',
    }


def main():
    from model import make_generator
    from train import ClearPatches, TemporalPairs, choose_device

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
    ids = ([str(path.relative_to(a.data)) for path in dataset.paths] if c['training'] == 'synthetic' else
           [row['input'] for row in dataset.rows])
    scores = evaluate_with_baseline(model, loader, device, c['training'], a.split, ids)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
