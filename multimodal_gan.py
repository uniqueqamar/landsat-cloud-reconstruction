"""Train and evaluate a compact SAR-guided GAN for optical cloud reconstruction."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio
import torch
from torch import nn
from torch.nn import functional as F

from compare_models import display
from evaluate_ground_detail import InteriorDetail
from evaluate_m3r import read_optical
from m3r_pilot import load_example, radar_on_grid
from model import Discriminator
from prediction_checks import validate_reconstruction_output
from quality_metrics import MaskedQuality


TOLERANCE = .02
ADV_WEIGHT = .002
GRADIENT_WEIGHT = .10


class FusionGenerator(nn.Module):
    """Fuse separate optical and coarse SAR streams on the optical grid."""

    def __init__(self):
        super().__init__()
        self.optical = nn.Sequential(nn.Conv2d(5, 24, 5, padding=2), nn.SiLU())
        self.radar = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(12, 12, 3, padding=1), nn.SiLU(),
        )
        self.fusion = nn.Sequential(nn.Conv2d(36, 24, 3, padding=1), nn.SiLU())
        self.context = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=4, dilation=4), nn.SiLU(),
        )
        self.head = nn.Conv2d(24, 4, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, mean):
        optical = self.optical(features[:, :5])
        radar = F.interpolate(self.radar(features[:, 5:]), size=optical.shape[-2:],
                              mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat((optical, radar), dim=1))
        return (mean + .12 * torch.tanh(self.head(fused + self.context(fused)))).clamp(0, 1)


@dataclass
class Sample:
    identifier: str
    observed: np.ndarray
    target: np.ndarray | None
    mask: np.ndarray
    radar: np.ndarray
    profile: dict | None = None

    def inputs(self, guided):
        visible = ~self.mask & np.isfinite(self.observed).all(axis=0)
        if not visible.any():
            raise ValueError(f'{self.identifier}: no valid visible context')
        mean = self.observed[:, visible].mean(axis=1)[:, None, None]
        centered = np.where(visible, (self.observed - mean) / .1, 0)
        radar = self.radar if guided else np.zeros_like(self.radar)
        features = np.concatenate((centered, ~visible[None], radar), axis=0)
        return torch.from_numpy(features[None]), torch.from_numpy(mean[None])


def samples(root, dataset, split, rng=None):
    if dataset == 'm3r':
        path, profile, features, _, observed, mask = load_example(root, split)
        target, _ = read_optical(path)
        yield Sample(path.stem, observed, np.clip(target, 0, 10000) / 10000, mask,
                     features['sar_guided'][0, 5:].numpy(), profile)
        return
    files = sorted((root / split).glob('*.npz'))
    if not files:
        raise ValueError(f'No prepared SEN12MS-CR samples in {root / split}')
    if rng is not None:
        rng.shuffle(files)
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            yield Sample(path.stem, data['observed'], data['target'], data['mask'], data['radar'])


def training_crop(sample, rng, guided):
    features, mean = sample.inputs(guided)
    mask = sample.mask & np.isfinite(sample.target).all(axis=0)
    if not mask.any():
        raise ValueError(f'{sample.identifier}: no valid training targets under cloud')
    # Sample a crop around a masked pixel and keep the whole-patch visible mean.
    rows, cols = np.nonzero(mask)
    index = rng.integers(len(rows))
    height, width = mask.shape
    size = min(96, height, width)
    y = int(np.clip(rows[index] - size // 2, 0, height - size))
    x = int(np.clip(cols[index] - size // 2, 0, width - size))
    window = np.s_[..., y:y + size, x:x + size]
    visible = np.where(np.isfinite(sample.observed), sample.observed, 0)
    target = np.where(mask, sample.target, visible)
    tensors = [features[window], torch.from_numpy(mask[None, None][window]),
               torch.from_numpy(visible[None][window]), torch.from_numpy(target[None][window])]
    rotation = int(rng.integers(4))
    tensors = [torch.rot90(value, rotation, (-2, -1)) for value in tensors]
    return (*tensors, mean)


def masked_loss(prediction, target, mask):
    residual = (prediction - target).abs().mean(dim=1, keepdim=True)
    return (residual * mask).sum() / mask.sum().clamp_min(1)


def gradient_loss(prediction, target, mask):
    total = prediction.new_zeros(())
    count = mask.new_zeros((), dtype=torch.int32)
    residual = prediction - target
    for axis in (-1, -2):
        length = mask.shape[axis] - 1
        valid = mask.narrow(axis, 0, length) & mask.narrow(axis, 1, length)
        total += (residual.diff(dim=axis).abs() * valid).sum()
        count += valid.sum()
    return total / (4 * count).clamp_min(1)


@torch.inference_mode()
def validation_mae(model, root, dataset, guided):
    quality = MaskedQuality()
    for sample in samples(root, dataset, 'validation'):
        prediction = model(*sample.inputs(guided))[0].numpy()
        selected = sample.mask & np.isfinite(sample.target).all(axis=0)
        quality.add(prediction, sample.target, selected)
    return quality.result()['masked_mae']


def train(args):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    generator = FusionGenerator()
    discriminator = Discriminator()
    g_opt = torch.optim.Adam(generator.parameters(), lr=3e-4, betas=(.5, .999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(.5, .999))
    adversarial = nn.BCEWithLogitsLoss()
    guided = not args.optical_only
    # Keep M3R's single example in memory. Stream SEN12 samples from disk.
    example = next(samples(args.data, args.dataset, 'train')) if args.dataset == 'm3r' else None

    args.output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    history = []
    train_ids = set()
    adversarial_steps = 0
    for epoch in range(args.epochs):
        totals = np.zeros(4)
        epoch_samples = (example for _ in range(args.steps_per_epoch)) if example is not None else samples(
            args.data, args.dataset, 'train', rng=rng)
        use_gan = epoch >= args.warmup
        for sample in epoch_samples:
            train_ids.add(sample.identifier)
            features, mask, observed, target, mean = training_crop(sample, rng, guided)
            reconstruction = generator(features, mean)
            candidate = torch.where(mask, reconstruction, observed)
            gan_loss = reconstruction.new_zeros(())
            if use_gan:
                adversarial_steps += 1
                d_opt.zero_grad()
                condition = torch.where(mask, 0., observed)
                real = discriminator(condition, mask.float(), target)
                fake = discriminator(condition, mask.float(), candidate.detach())
                d_loss = .5 * (adversarial(real, torch.ones_like(real)) +
                               adversarial(fake, torch.zeros_like(fake)))
                d_loss.backward()
                d_opt.step()
                discriminator.requires_grad_(False)
                score = discriminator(condition, mask.float(), candidate)
                gan_loss = adversarial(score, torch.ones_like(score))
            g_opt.zero_grad()
            mae = masked_loss(reconstruction, target, mask)
            detail = gradient_loss(reconstruction, target, mask)
            loss = mae + GRADIENT_WEIGHT * detail + ADV_WEIGHT * gan_loss
            loss.backward()
            g_opt.step()
            discriminator.requires_grad_(True)
            totals += (mae.item(), detail.item(), gan_loss.item(), 1)
        validation = validation_mae(generator, args.data, args.dataset, guided) if args.dataset == 'sen12' else None
        record = {'epoch': epoch + 1, 'steps': int(totals[3]), 'adversarial_steps': adversarial_steps,
                  'training_mae': totals[0] / totals[3], 'training_gradient_mae': totals[1] / totals[3],
                  'training_adversarial_loss': totals[2] / totals[3], 'validation_mae': validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if use_gan and (validation is None or validation < best):
            best = validation if validation is not None else best
            torch.save({'model': generator.state_dict(), 'dataset': args.dataset, 'guided': guided,
                        'train_ids': sorted(train_ids), 'epoch': epoch + 1,
                        'adversarial_steps': adversarial_steps, 'seed': 42}, args.output / 'best.pt')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')


def save_preview(path, sample, prediction, mean):
    fill = np.where(sample.mask, mean, sample.observed)
    panels = [('Cloudy optical', sample.observed), ('Mean fill', fill),
              ('GAN reconstruction', prediction), ('Clear reference', sample.target)]
    image = Image.new('RGB', (988, 302), '#f4f5f6')
    draw = ImageDraw.Draw(image)
    for i, (label, bands) in enumerate(panels):
        x = 20 + 244 * i
        draw.text((x, 12), label, fill='#18242f')
        image.paste(display(np.nan_to_num(bands)), (x, 35))
    draw.text((20, 275), f'Held-out patch {sample.identifier} | same display scale in each panel.', fill='#18242f')
    image.save(path)


def load_weights(path, dataset):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if checkpoint['dataset'] != dataset or checkpoint['adversarial_steps'] <= 0:
        raise ValueError('Expected adversarially trained weights for this dataset')
    model = FusionGenerator().eval()
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint


def reconstructed(model, guided, sample):
    features, mean = sample.inputs(guided)
    candidate = model(features, mean)[0].numpy()
    prediction = np.where(sample.mask, candidate, sample.observed)
    if not np.isfinite(prediction[:, sample.mask]).all():
        raise ValueError('Model produced invalid values inside the reconstruction mask')
    if np.isfinite(sample.observed).all():
        validate_reconstruction_output(prediction, sample.observed, sample.mask)
    else:
        np.testing.assert_array_equal(prediction[:, ~sample.mask], sample.observed[:, ~sample.mask])
    return prediction, mean[0].numpy()


def write_prediction(path, sample, prediction, dataset):
    profile = {**sample.profile, 'dtype': 'float32', 'nodata': np.nan, 'compress': 'deflate'}
    with rasterio.open(path, 'w', **profile) as output:
        output.write(prediction)
        output.descriptions = ('blue', 'green', 'red', 'near infrared')
        output.update_tags(dataset=dataset, covered_pixels='GAN estimates under input mask')


@torch.inference_mode()
def predict(args):
    model, checkpoint = load_weights(args.weights, args.dataset)
    raw, grid = read_optical(args.cloudy)
    with rasterio.open(args.cloudy) as source:
        profile = source.profile
    mask = np.load(args.mask, allow_pickle=False)
    if mask.shape != raw.shape[1:] or not np.isin(mask, (0, 1)).all():
        raise ValueError('Expected binary mask on the cloudy optical grid')
    radar = np.zeros((3, *mask.shape), dtype=np.float32)
    if checkpoint['guided']:
        if args.sar is None:
            raise ValueError('SAR-guided checkpoints require --sar')
        values, valid = radar_on_grid(args.sar, grid)
        radar = np.concatenate((values, valid), axis=0)
    observed = np.clip(raw, 0, 10000) / 10000
    sample = Sample(args.cloudy.stem, observed, None, mask.astype(bool), radar, profile)
    prediction, _ = reconstructed(model, checkpoint['guided'], sample)
    args.output.mkdir(parents=True, exist_ok=True)
    write_prediction(args.output / 'prediction.tif', sample, prediction, args.dataset)
    np.save(args.output / 'estimated_pixels.npy', sample.mask, allow_pickle=False)


@torch.inference_mode()
def evaluate(args):
    model, checkpoint = load_weights(args.weights, args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    quality = {name: MaskedQuality() for name in ('gan', 'mean_fill')}
    detail = {name: InteriorDetail() for name in quality}
    correct = {name: 0 for name in quality}
    patch_ids = []
    scenes = {}
    for sample in samples(args.data, args.dataset, 'test'):
        if sample.identifier in checkpoint['train_ids']:
            raise ValueError(f'{sample.identifier}: test patch was used in training')
        prediction, mean = reconstructed(model, checkpoint['guided'], sample)
        patch_ids.append(sample.identifier)
        valid = sample.mask & np.isfinite(sample.target).all(axis=0)
        padded = F.pad(torch.from_numpy((~valid)[None, None]).float(), (4, 4, 4, 4), value=1)
        interior = (F.max_pool2d(padded, 9, stride=1) == 0)[0, 0].numpy()
        for name, values in [('gan', prediction), ('mean_fill', np.where(sample.mask, mean, sample.observed))]:
            quality[name].add(values, sample.target, valid)
            detail[name].add(values, sample.target, interior)
            hits = int((np.abs(values[:, valid] - sample.target[:, valid]) <= TOLERANCE).all(axis=0).sum())
            correct[name] += hits
            if args.dataset == 'sen12':
                scene = sample.identifier.rsplit('_', 1)[0]
                by_method = scenes.setdefault(scene, {})
                score, count = by_method.setdefault(name, (MaskedQuality(), 0))
                score.add(values, sample.target, valid)
                by_method[name] = score, count + hits
        if len(patch_ids) == 1:
            save_preview(args.output / 'comparison.png', sample, prediction, mean)
        if sample.profile is not None:
            write_prediction(args.output / f'{sample.identifier}.tif', sample, prediction, args.dataset)
    results = {}
    for name, metric in quality.items():
        results[name] = dict(**metric.result(),
                             fraction_within_tolerance=correct[name] / metric.pixels,
                             interior_detail=detail[name].result())
    report = {'dataset': args.dataset, 'sar_guided': checkpoint['guided'], 'selected_epoch': checkpoint['epoch'],
              'adversarial_steps_at_checkpoint': checkpoint['adversarial_steps'], 'test_patches': patch_ids,
              'all_four_bands_absolute_tolerance': TOLERANCE, 'results': results}
    if scenes:
        report['by_scene'] = {
            scene: {
                name: dict(**metric.result(), fraction_within_tolerance=count / metric.pixels)
                for name, (metric, count) in methods.items()
            }
            for scene, methods in scenes.items()
        }
    (args.output / 'scores.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'evaluate', 'predict'))
    parser.add_argument('--dataset', choices=('m3r', 'sen12'), required=True)
    parser.add_argument('--data', type=Path, help='Dataset root for training or evaluation')
    parser.add_argument('--cloudy', type=Path, help='Four-band optical DN GeoTIFF for prediction')
    parser.add_argument('--mask', type=Path, help='Binary NPY mask on the optical grid')
    parser.add_argument('--sar', type=Path, help='Two-band VV/VH SAR GeoTIFF in dB')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--optical-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps-per-epoch', type=int, default=30, help='M3R public example only')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == 'predict':
        if not all((args.weights, args.cloudy, args.mask)):
            parser.error('Prediction requires --weights, --cloudy, and --mask')
        predict(args)
        return
    if args.data is None:
        parser.error('Training and evaluation require --data')
    if args.mode == 'evaluate':
        if args.weights is None:
            parser.error('Evaluation requires --weights')
        evaluate(args)
    else:
        if not 0 <= args.warmup < args.epochs or args.steps_per_epoch < 1:
            parser.error('Specify a training schedule with at least one adversarial epoch')
        train(args)


if __name__ == '__main__':
    main()
