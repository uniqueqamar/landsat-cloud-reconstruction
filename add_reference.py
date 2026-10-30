"""Fetch small aligned temporal-reference windows from a public Landsat mirror.

The eight-day temporal gap means these targets are proxy observations, not
simultaneous ground truth. Tokens are used in memory and never packaged.
"""
import argparse
import csv
import json
import urllib.parse
import urllib.request
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

BANDS = ['blue', 'green', 'red', 'nir08', 'qa_pixel', 'qa_radsat']
BAD = sum(1 << b for b in (0, 1, 2, 3, 4, 5))
SAT = sum(1 << b for b in (1, 2, 3, 4, 11))


def sign(href):
    url = 'https://planetarycomputer.microsoft.com/api/sas/v1/sign?href=' + urllib.parse.quote(href, safe='')
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)['href']


def add(root, catalog, signed=None, reference_dir=None):
    item = json.loads(catalog.read_text())['features'][0]
    assets = item['assets']
    urls = ([str(reference_dir / (k + '.tif')) for k in BANDS] if reference_dir else
            [signed['assets'][k]['signed_href'] if signed else sign(assets[k]['href']) for k in BANDS])
    rows = list(csv.DictReader((root / 'manifest.csv').open()))
    rows = [r for r in rows if r['category'] == 'cloudy']
    pairs, rejected = [], []
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.TIF',
               GDAL_HTTP_CONNECTTIMEOUT=20, GDAL_HTTP_TIMEOUT=60, GDAL_CACHEMAX=64)
    with rasterio.Env(**env), ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(u)) for u in urls]
        print('Opened six reference assets', flush=True)
        for n, row in enumerate(rows):
            source = root / row['path']
            with rasterio.open(source) as cloudy:
                arrays = []
                for i, src in enumerate(sources):
                    with WarpedVRT(src, crs=cloudy.crs, transform=cloudy.transform,
                                   width=cloudy.width, height=cloudy.height,
                                   resampling=Resampling.bilinear if i < 4 else Resampling.nearest,
                                   src_nodata=0 if i < 4 else None,
                                   nodata=0 if i < 4 else 1) as vrt:
                        arrays.append(vrt.read(1))
                profile = cloudy.profile.copy()
            target = np.stack(arrays).astype('uint16')
            if np.any(target[4] & BAD) or np.any(target[5] & SAT) or np.any(target[:4] == 0):
                rejected.append(row['path'])
                continue
            relative = Path('data/reference') / row['split'] / source.name
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(target)
                for i, name in enumerate(BANDS, 1):
                    dst.set_band_description(i, name)
                dst.update_tags(reference_scene=item['id'], reference_date='2026-10-05',
                                temporal_gap_days='8', reference_role='temporal proxy target',
                                optical_resampling='bilinear', quality_resampling='nearest')
            pairs.append(dict(input=row['path'], target=relative.as_posix(), split=row['split'],
                              input_date='2026-09-27', target_date='2026-10-05', gap_days=8,
                              cloud_fraction=row['cloud_fraction']))
            print(f'Reference {n + 1}/{len(rows)}: {row["split"]}/{source.name}', flush=True)
    with (root / 'pairs.csv').open('w', newline='') as f:
        fields = ['input', 'target', 'split', 'input_date', 'target_date', 'gap_days', 'cloud_fraction']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pairs)
    report = dict(reference_scene=item['id'], date='2026-10-05', gap_days=8,
                  accepted_pairs=len(pairs), rejected_inputs=rejected,
                  caveat='Landsat 8 temporal proxy targets for Landsat 9 inputs; seasonal or real changes may occur.')
    (root / 'reference_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--signed-catalog', type=Path, help='Internal temporary cache; never distribute signed URLs')
    p.add_argument('--reference-dir', type=Path, help='Optional local mirror bands named blue.tif, green.tif, etc.')
    a = p.parse_args()
    add(a.data, a.catalog, json.loads(a.signed_catalog.read_text()) if a.signed_catalog else None,
        a.reference_dir)
