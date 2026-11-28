"""Compare VV and VH darkening on the matched Porto Alegre radar windows."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio

from radar_flood import COLORS, geojson_mask, overlap, read_backscatter


METHODS = (
    ('vv', 'VV darkening'),
    ('vh', 'VH darkening'),
    ('either', 'Either band darkens'),
    ('both', 'Both bands darken'),
)
MINIMUM_DROP_DB = 3


def compare(inputs):
    grid = None
    before, event = {}, {}
    for band, suffix in (('vv', ''), ('vh', '_vh')):
        for label, images in (('before', before), ('after', event)):
            data, current_grid = read_backscatter(inputs / f'{label}{suffix}.tif')
            if grid is not None and current_grid != grid:
                raise ValueError('All four radar windows must share the same grid')
            grid = current_grid
            images[band] = data
    valid = np.logical_and.reduce([np.isfinite(band) for band in (*before.values(), *event.values())])
    vv = valid & (event['vv'] - before['vv'] <= -MINIMUM_DROP_DB)
    vh = valid & (event['vh'] - before['vh'] <= -MINIMUM_DROP_DB)
    candidates = {'vv': vv, 'vh': vh, 'either': vv | vh, 'both': vv & vh}
    return event, grid, valid, candidates


def render(path, event, valid, candidates):
    def radar_image(band):
        shade = np.rint(np.clip((np.nan_to_num(band, nan=-32) + 32) / 32, 0, 1) * 255).astype('uint8')
        return np.repeat(shade[..., None], 3, axis=-1)

    vv = radar_image(event['vv'])
    panels = [('Event radar | VV', vv), ('Event radar | VH', radar_image(event['vh']))]
    for name, label in METHODS:
        overlay = vv.copy()
        overlay[candidates[name]] = COLORS[1]
        overlay[~valid] = COLORS[255]
        panels.append((label, overlay))

    width = 350
    height = round(width * valid.shape[0] / valid.shape[1])
    canvas = Image.new('RGB', (3 * (width + 12) + 12, 2 * (height + 36) + 65), '#f4f5f6')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 11), 'Porto Alegre flood | 11 May 2024 | Sentinel-1 track 24', fill='#18242f')
    for i, (label, panel) in enumerate(panels):
        x = 12 + i % 3 * (width + 12)
        y = 37 + i // 3 * (height + 36)
        draw.text((x, y), label, fill='#18242f')
        canvas.paste(Image.fromarray(panel).resize((width, height), Image.Resampling.NEAREST), (x, y + 18))
    y = 2 * (height + 36) + 26
    draw.rectangle((12, y, 24, y + 12), fill=COLORS[1])
    draw.text((31, y), 'At least 3 dB darker than baseline', fill='#18242f')
    draw.rectangle((260, y, 272, y + 12), fill=COLORS[255])
    draw.text((279, y), 'Unknown', fill='#18242f')
    draw.text((12, y + 18), 'Change layers are overlaid on event VV. Radar views share a −32 to 0 dB display scale.', fill='#18242f')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='Crops from fetch_porto_alegre.py --both-bands')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    event, grid, valid, candidates = compare(args.inputs)
    reference = geojson_mask(args.inputs / 'reference.geojson', grid)
    scored = valid & geojson_mask(args.inputs / 'aoi.geojson', grid)
    if not (reference & scored).any():
        parser.error('No mapped flood pixels overlap the common valid radar grid')
    provenance = {
        'before_time_utc': '2023-11-25T08:49:46.038313Z',
        'event_time_utc': '2024-05-11T08:49:55.541452Z',
        'reference_time_utc': '2024-05-10T22:00:00Z',
        'relative_orbit': 24,
        'minimum_drop_db': MINIMUM_DROP_DB,
    }
    report = {
        **provenance,
        'common_valid_pixels': int(valid.sum()),
        'scored_pixels': int(scored.sum()),
        'reference_flood_pixels': int((reference & scored).sum()),
        'methods': {name: overlap(candidate, reference, scored) for name, candidate in candidates.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    crs, transform, width, height = grid
    with rasterio.open(
        args.output / 'darkening_by_band.tif', 'w', driver='GTiff', count=len(METHODS),
        dtype='uint8', nodata=255, crs=crs, transform=transform, width=width,
        height=height, compress='deflate',
    ) as destination:
        for index, (name, label) in enumerate(METHODS, 1):
            destination.write(np.where(valid, candidates[name], 255).astype('uint8'), index)
            destination.set_band_description(index, label)
        destination.update_tags(**provenance, classes='0: no threshold crossing, 1: darkened, 255: unknown')
    render(args.output / 'comparison.png', event, valid, candidates)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
