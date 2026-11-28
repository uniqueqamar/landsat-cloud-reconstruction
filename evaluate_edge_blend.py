"""Tune edge blending on validation, then evaluate the fixed setting on test."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from edge_blend import edge_blend
from model import make_generator
from quality_metrics import MaskedQuality
from spatial_fill import spatial_fill
from train import TemporalPairs


def audit(root, split, models, settings):
    data = TemporalPairs(root, split)
    keys = [(name, width, strength) for name in models for width, strength in settings]
    quality = {key: MaskedQuality() for key in keys}
    boundaries = {key: np.zeros(3) for key in keys}
    for index, (image, mask, _) in enumerate(data):
        with rasterio.open(root / data.rows[index]['target']) as source:
            reference = source.read([1, 2, 3, 4]).astype('float32') * .0000275 - .2
        image, mask = image[None], mask[None]
        interpolated = spatial_fill(image, mask)
        covered = mask[0, 0].numpy().astype(bool)
        crossings = [np.diff(covered, axis=axis) for axis in (0, 1)]
        reference_gradients = [
            np.diff(reference, axis=axis + 1)[:, crossed]
            for axis, crossed in enumerate(crossings)
        ]
        for name, model in models.items():
            raw = model(image, mask)
            for width, strength in settings:
                key = (name, width, strength)
                prediction = raw
                if width:
                    prediction = edge_blend(image, mask, raw, width, strength, interpolated=interpolated)
                out = prediction[0].numpy()
                quality[key].add(out, reference, covered)
                for axis, crossed in enumerate(crossings):
                    gradient = np.diff(out, axis=axis + 1)[:, crossed]
                    boundaries[key] += (
                        np.abs(gradient - reference_gradients[axis]).sum(),
                        np.abs(gradient).sum(),
                        gradient.size,
                    )
    results = {}
    for key, score in quality.items():
        name, width, strength = key
        error, jump, count = boundaries[key]
        results[f'{name}:{width}:{strength}'] = dict(
            **score.result(),
            boundary_gradient_mae=float(error / count) if count else None,
            mean_boundary_jump=float(jump / count) if count else None,
            boundary_band_value_count=int(count),
        )
    return {'split': split, 'patches': len(data), 'results': results}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(2)
    models = {}
    for name, path in (
        ('srgan', 'runs/srgan_cloud_types/control.pt'),
        ('v2', 'runs/context_residual_v2/best.pt'),
    ):
        checkpoint = torch.load(root / path, weights_only=True, map_location='cpu')
        model = make_generator(checkpoint).eval()
        model.load_state_dict(checkpoint['model'])
        models[name] = model
    baseline = (0, 0)
    candidates = [(width, strength) for width in (2, 4, 8, 16) for strength in (.5, 1.)]
    validation = audit(root, 'val', models, [baseline, *candidates])
    selected = min(candidates, key=lambda pair: validation['results'][f'srgan:{pair[0]}:{pair[1]}']['masked_mae'])
    test = audit(root, 'test', models, [baseline, selected])
    args.output.mkdir(parents=True, exist_ok=True)
    for result in (validation, test):
        result['selected'] = selected
        (args.output / f"{result['split']}.json").write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
