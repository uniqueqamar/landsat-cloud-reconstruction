"""Measure ground detail inside real and controlled cloud masks on the Landsat test split."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from compare_models import display
from edge_blend import edge_blend
from model import make_generator
from prediction_checks import validate_reconstruction_output
from quality_metrics import MaskedQuality
from spatial_fill import spatial_fill
from train import TemporalPairs


class InteriorDetail:
    """Score signed reflectance gradients between neighboring interior pixels."""

    def __init__(self):
        self.values = np.zeros(6, dtype=np.float64)

    def add(self, prediction, reference, interior):
        for axis in (0, 1):
            along = np.moveaxis(interior, axis, -1)
            pairs = along[..., 1:] & along[..., :-1]
            pairs = np.moveaxis(pairs, -1, axis)
            estimate = np.diff(prediction, axis=axis + 1)[:, pairs].astype(np.float64)
            truth = np.diff(reference, axis=axis + 1)[:, pairs].astype(np.float64)
            self.values += (
                estimate.size, np.abs(estimate - truth).sum(), np.vdot(estimate, truth),
                np.square(estimate).sum(), np.square(truth).sum(), np.abs(truth).sum(),
            )

    def result(self):
        count, error, dot, estimated_energy, reference_energy, mean_baseline_error = self.values
        return {
            'gradient_values': int(count),
            'gradient_mae': float(error / count) if count else None,
            'zero_gradient_baseline_mae': float(mean_baseline_error / count) if count else None,
            'gradient_cosine': float(dot / np.sqrt(estimated_energy * reference_energy))
            if estimated_energy and reference_energy else None,
            'gradient_rms_ratio': float(np.sqrt(estimated_energy / reference_energy))
            if reference_energy else None,
        }


def write_preview(output, observed, predictions, reference, patch):
    panels = [
        ('Known ground hidden for test', observed),
        ('Spatial interpolation', predictions['spatial_fill']),
        ('SRGAN and edge blend', predictions['srgan_blended']),
        ('Known clear ground', reference),
    ]
    canvas = Image.new('RGB', (988, 301), '#f5f5f1')
    draw = ImageDraw.Draw(canvas)
    for index, (label, bands) in enumerate(panels):
        x = 20 + index * 244
        draw.text((x, 14), label, fill='#172f30')
        canvas.paste(display(bands), (x, 37))
    draw.text((20, 277), f'Controlled test on {patch}. Same RGB display scale in each panel.', fill='#374e50')
    canvas.save(output)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    torch.set_num_threads(2)
    checkpoint = torch.load(root / 'runs/srgan_cloud_types/control.pt', weights_only=True, map_location='cpu')
    model = make_generator(checkpoint).eval()
    model.load_state_dict(checkpoint['model'])
    dataset = TemporalPairs(root, 'test')
    experiments = {}
    for mode in ('real_cloud', 'known_ground_hidden'):
        metrics = {}
        for index, (image, mask, _) in enumerate(dataset):
            row = dataset.rows[index]
            with rasterio.open(root / row['target']) as source:
                reference = source.read([1, 2, 3, 4]).astype('float32') * .0000275 - .2
            image, mask = image[None], mask[None]
            hidden = mask[0, 0].numpy().astype(bool)
            # A 9×9 neighborhood must be fully masked. Outside the patch is
            # excluded so the statistic only uses known interior neighborhoods.
            clear = F.pad(1 - mask, (4, 4, 4, 4), value=1)
            interior = (F.max_pool2d(clear, 9, stride=1) == 0)[0, 0].numpy()
            if mode == 'known_ground_hidden':
                image = torch.where(mask.bool(), .65, torch.from_numpy(reference)[None])
            observed = image[0].numpy()
            mean = image[:, :, ~hidden].mean(dim=-1)[..., None, None]
            spatial = spatial_fill(image, mask)
            raw = model(image, mask)
            blended = edge_blend(image, mask, raw, width=16, strength=.5, interpolated=spatial)
            estimates = {
                'mean_fill': torch.where(mask.bool(), mean, image),
                'spatial_fill': spatial, 'srgan': raw, 'srgan_blended': blended,
            }
            outputs = {name: value[0].numpy() for name, value in estimates.items()}
            for name, output in outputs.items():
                validate_reconstruction_output(output, observed, hidden)
                if name not in metrics:
                    metrics[name] = MaskedQuality(), InteriorDetail()
                quality, detail = metrics[name]
                quality.add(output, reference, interior)
                detail.add(output, reference, interior)
            if mode == 'known_ground_hidden' and index == 0:
                write_preview(args.output / 'comparison.png', observed, outputs, reference, Path(row['input']).stem)
        experiments[mode] = {
            name: {'interior_reflectance': quality.result(), 'interior_detail': detail.result()}
            for name, (quality, detail) in metrics.items()
        }
    result = {'split': 'test', 'patches': len(dataset), 'masked_neighborhood_width': 9, 'experiments': experiments}
    (args.output / 'scores.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
