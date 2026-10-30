"""Run a saved generator on one six-band patch, preserving observed clear pixels."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
import torch
from model import Generator
from train import choose_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    device = choose_device()
    checkpoint = torch.load(a.weights, map_location='cpu', weights_only=True)
    model = Generator(checkpoint['width']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with rasterio.open(a.input) as src:
        if src.count != 6:
            raise ValueError('Expected prepared patch with B2,B3,B4,B5,QA_PIXEL,QA_RADSAT')
        dn, qa, sat = src.read([1, 2, 3, 4]), src.read(5), src.read(6)
        if src.height % 8 or src.width % 8:
            raise ValueError('Patch dimensions must be divisible by eight')
        if np.any(qa & 1) or np.any(dn == 0) or np.any(sat & sum(1 << b for b in (1, 2, 3, 4, 11))):
            raise ValueError('Patch contains invalid or saturated observations')
        profile = src.profile.copy()
    image = dn.astype('float32') * .0000275 - .2
    mask = ((qa & sum(1 << b for b in (1, 2, 3, 4))) != 0).astype('float32')
    with torch.no_grad():
        out = model(torch.from_numpy(image[None]).to(device),
                    torch.from_numpy(mask[None, None]).to(device)).cpu().numpy()[0]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=4, dtype='float32')
    with rasterio.open(a.output, 'w', **profile) as dst:
        dst.write(out)
        for i, band in enumerate(['blue', 'green', 'red', 'nir'], 1):
            dst.set_band_description(i, band + '_estimated_surface_reflectance')
        dst.update_tags(method=f'{checkpoint["training"]}-trained generator; experimental reconstruction',
                        mask_source='QA_PIXEL cloud/cirrus/dilated cloud/shadow',
                        warning='Generated pixels are estimates, not ground truth')
    rgb = np.moveaxis(out[[2, 1, 0]], 0, -1)
    Image.fromarray((np.clip(rgb / .35, 0, 1) ** (1 / 1.8) * 255).astype('uint8')).save(a.output.with_suffix('.png'))
    np.save(a.output.with_suffix('.mask.npy'), mask.astype('uint8'))
    print(f'Saved experimental reconstruction: {a.output}')


if __name__ == '__main__':
    main()
