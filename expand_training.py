"""Add bounded training-only shards; preserve the original validation/test sets."""
import argparse
import csv
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from prepare_data import FILES, BAD_CLEAR, CLOUD, SAT, split_for, SCENE


def expand(root, raw, reference, limit=96):
    dest = root / 'expanded_train'
    if dest.exists():
        raise FileExistsError('expanded_train already exists; use a clean destination')
    existing = list(csv.DictReader((root / 'manifest.csv').open()))
    used = {(int(r['row']), int(r['col'])) for r in existing}
    refs = ['blue', 'green', 'red', 'nir08', 'qa_pixel', 'qa_radsat']
    samples, records = [], []
    with ExitStack() as stack:
        inputs = [stack.enter_context(rasterio.open(raw / p)) for p in FILES]
        targets = [stack.enter_context(rasterio.open(reference / (p + '.tif'))) for p in refs]
        base = inputs[0]
        aligned = [stack.enter_context(WarpedVRT(s, crs=base.crs, transform=base.transform,
                    width=base.width, height=base.height,
                    resampling=Resampling.bilinear if i < 4 else Resampling.nearest,
                    src_nodata=0 if i < 4 else None, nodata=0 if i < 4 else 1))
                   for i, s in enumerate(targets)]
        coordinates = [(y, x) for y in range(0, base.height-127, 128)
                       for x in range(0, base.width-127, 128)
                       if split_for(y//128, x//128) == 'train' and (y, x) not in used]
        np.random.default_rng(2048).shuffle(coordinates)
        for y, x in coordinates:
            window = Window(x, y, 128, 128)
            qa, sat = [s.read(1, window=window) for s in inputs[4:]]
            frac = float(np.mean((qa & CLOUD) != 0))
            if np.any(qa & ((1 << 0) | (1 << 5))) or np.any(sat & SAT) or not .15 <= frac <= .75:
                continue
            tqa, tsat = [s.read(1, window=window) for s in aligned[4:]]
            if np.any(tqa & BAD_CLEAR) or np.any(tsat & SAT):
                continue
            observed = np.stack([s.read(1, window=window) for s in inputs[:4]])
            target = np.stack([s.read(1, window=window) for s in aligned[:4]])
            if np.any(observed == 0) or np.any(target == 0):
                continue
            samples.append((observed, target, qa))
            bounds = rasterio.windows.bounds(window, base.transform)
            records.append(dict(row=y, col=x, split='train', cloud_fraction=frac,
                                left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3]))
            if len(samples) >= limit:
                break
    dest.mkdir()
    shard_rows = []
    for start in range(0, len(samples), 8):
        group = samples[start:start+8]
        name = f'train_{start//8:03d}.npz'
        np.savez_compressed(dest / name,
                            observed=np.stack([g[0] for g in group]),
                            target=np.stack([g[1] for g in group]),
                            qa_pixel=np.stack([g[2] for g in group]))
        for i, record in enumerate(records[start:start+8]):
            shard_rows.append(dict(shard='expanded_train/' + name, index=i, **record))
    with (dest / 'index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(shard_rows[0]))
        w.writeheader(); w.writerows(shard_rows)
    summary = dict(additional_training_pairs=len(samples), patch_size=128,
                   source_scene=SCENE, reference_scene='LC08_L2SP_141044_20261005_20261009_02_T1',
                   gap_days=8, added_validation_pairs=0, added_test_pairs=0,
                   selection='Deterministic training macroblocks only; exclude all existing patch coordinates',
                   crs=str(base.crs), optical_scale=.0000275, optical_offset=-.2,
                   optical_resampling='bilinear', quality_resampling='nearest',
                   shard_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.npz')})
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != 'shard_hashes'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--limit', type=int, default=96)
    a = p.parse_args(); expand(a.data, a.raw, a.reference, a.limit)
