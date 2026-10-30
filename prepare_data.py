"""Build a bounded, georeferenced starter dataset from six local Landsat rasters."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

SCENE = 'LC09_L2SP_141044_20260927_20260928_02_T1'
FILES = ['B2.tif', 'B3.tif', 'B4.tif', 'B5.tif', 'qa_pixel.tif', 'QA_RADSAT.tif']
BAD_CLEAR = sum(1 << b for b in (0, 1, 2, 3, 4, 5))
CLOUD = sum(1 << b for b in (1, 2, 3))
SAT = sum(1 << b for b in (1, 2, 3, 4, 11))


def split_for(tile_row, tile_col):
    # One unused tile along each macroblock edge separates different splits.
    if tile_row % 8 in (0, 7) or tile_col % 8 in (0, 7):
        return None
    key = f'{SCENE}:{tile_row // 8}:{tile_col // 8}:42'
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    return 'train' if bucket < 7 else ('val' if bucket == 7 else 'test')


def build(raw, output, size=128):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.csv').exists():
        raise FileExistsError('Use a fresh output directory to avoid stale patch files.')
    counts = Counter()
    candidates = []
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(raw / p)) for p in FILES]
        base = sources[0]
        for src in sources[1:]:
            assert (src.width, src.height, src.crs, src.transform) == (
                base.width, base.height, base.crs, base.transform), 'Raster grids differ'
        for y in range(0, base.height - size + 1, size):
            for x in range(0, base.width - size + 1, size):
                split = split_for(y // size, x // size)
                if split is None:
                    continue
                window = Window(x, y, size, size)
                qa, sat = [src.read(1, window=window) for src in sources[4:]]
                if np.any(qa & 1) or np.any(sat & SAT):
                    continue
                cloud_fraction = float(np.mean((qa & CLOUD) != 0))
                if np.all((qa & BAD_CLEAR) == 0):
                    category = 'clear'
                elif .15 <= cloud_fraction <= .75 and not np.any(qa & (1 << 5)):
                    category = 'cloudy'
                else:
                    continue
                candidates.append((category, split, y, x, cloud_fraction))
                counts[f'candidate_{category}_{split}'] += 1
        rng = np.random.default_rng(42)
        rng.shuffle(candidates)
        caps = {'clear': {'train': 96, 'val': 24, 'test': 24},
                'cloudy': {'train': 24, 'val': 12, 'test': 12}}
        rows = []
        for category, split, y, x, fraction in candidates:
            key = f'selected_{category}_{split}'
            if counts[key] >= caps[category][split]:
                continue
            window = Window(x, y, size, size)
            dn = np.stack([src.read(1, window=window) for src in sources[:4]])
            if np.any(dn == 0):
                continue
            qa, sat = [src.read(1, window=window) for src in sources[4:]]
            name = f'data/{category}/{split}/r{y:05d}_c{x:05d}.tif'
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            profile = dict(driver='GTiff', height=size, width=size, count=6,
                           dtype='uint16', crs=base.crs,
                           transform=base.window_transform(window), compress='deflate')
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(np.concatenate([dn, qa[None], sat[None]], axis=0))
                for i, desc in enumerate(['SR_B2_blue', 'SR_B3_green', 'SR_B4_red',
                                          'SR_B5_nir', 'QA_PIXEL', 'QA_RADSAT'], 1):
                    dst.set_band_description(i, desc)
                dst.update_tags(source_scene=SCENE, target_available='false',
                                reflectance_scale='0.0000275', reflectance_offset='-0.2')
            b = rasterio.windows.bounds(window, base.transform)
            rows.append(dict(path=name, category=category, split=split,
                             row=y, col=x, cloud_fraction=f'{fraction:.6f}',
                             crs=str(base.crs), left=b[0], bottom=b[1], right=b[2], top=b[3]))
            counts[key] += 1
    if not rows:
        raise RuntimeError('No valid patches found')
    with (output / 'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = dict(scene=SCENE, date='2026-09-27', pixel_size_m=30,
                   patch_size=size, counts=dict(counts), real_clear_targets=False,
                   sar_available=False,
                   note='Clear means QA-screened. Real cloudy patches are unpaired.')
    (output / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--size', type=int, default=128)
    a = p.parse_args()
    build(a.raw, a.output, a.size)
