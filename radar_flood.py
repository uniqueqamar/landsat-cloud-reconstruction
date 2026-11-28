"""Map candidate new open water from an aligned Sentinel-1 RTC pair."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import ColorInterp
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


COLORS = {1: (28, 201, 240), 2: (56, 101, 197), 255: (102, 106, 112)}
CLASSES = {0: 'no open-water candidate', 1: 'new darkening candidate',
           2: 'already-dark surface', 255: 'unknown'}


def read_backscatter(path):
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f'{path}: missing CRS')
        if source.width * source.height > 4_000_000:
            raise ValueError(f'{path}: crop both images to a common window of at most 4 million pixels')
        if source.count != 1 and source.colorinterp != (ColorInterp.gray, ColorInterp.alpha):
            raise ValueError(f'{path}: expected one RTC polarization band, optionally with alpha')
        band = source.read(1, masked=True, out_dtype='float32')
        linear = band.filled(np.nan)
        if source.count == 2:
            linear[source.read(2) == 0] = np.nan
        valid = np.isfinite(linear) & (linear > 0)
        db = np.full(linear.shape, np.nan, dtype=np.float32)
        np.log10(linear, where=valid, out=db)
        db *= 10
        grid = source.crs, source.transform, source.width, source.height
    return db, grid


def classify(before, after, dark_db, drop_db):
    valid = np.isfinite(before) & np.isfinite(after)
    already_dark = valid & (before <= dark_db) & (after <= dark_db)
    change_only = valid & (after - before <= -drop_db)
    candidate = change_only & (before > dark_db) & (after <= dark_db)
    padded = np.pad(candidate, 1)
    votes = np.zeros(candidate.shape, dtype=np.uint8)
    height, width = candidate.shape
    for y in range(3):
        for x in range(3):
            votes += padded[y:y + height, x:x + width]
    classes = np.full(candidate.shape, 255, dtype=np.uint8)
    classes[valid] = 0
    classes[already_dark] = 2
    classes[candidate & (votes >= 5)] = 1
    return classes, change_only


def geojson_mask(path, grid):
    crs, transform, width, height = grid
    data = json.loads(path.read_text())
    if data.get('type') != 'FeatureCollection' or data.get('exceededTransferLimit'):
        raise ValueError(f'{path}: expected a complete WGS84 GeoJSON FeatureCollection')
    geometries = [
        transform_geom('EPSG:4326', crs, feature['geometry'])
        for feature in data['features'] if feature.get('geometry')
    ]
    if not geometries:
        raise ValueError(f'{path}: no reference geometries')
    return geometry_mask(geometries, (height, width), transform, invert=True)


def overlap(predicted, reference, valid):
    tp = int((predicted & reference & valid).sum())
    fp = int((predicted & ~reference & valid).sum())
    fn = int((~predicted & reference & valid).sum())
    return {
        'true_positive_pixels': tp, 'false_positive_pixels': fp, 'false_negative_pixels': fn,
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': tp / (tp + fn) if tp + fn else None,
        'iou': tp / (tp + fp + fn) if tp + fp + fn else None,
    }


def preview(path, before, after, classes, change_only, dates):
    def grayscale(band):
        shade = (np.clip((np.nan_to_num(band, nan=-28) + 28) / 28, 0, 1) * 255).astype('uint8')
        return np.repeat(shade[..., None], 3, axis=-1)

    panels = [grayscale(before), grayscale(after), grayscale(after), grayscale(after)]
    panels[2][change_only] = COLORS[1]
    panels[2][classes == 255] = COLORS[255]
    for code, color in COLORS.items():
        panels[-1][classes == code] = color
    height, width = classes.shape
    scale = min(1, 420 / max(height, width))
    panel_width, panel_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas = Image.new('RGB', (max(590, len(panels) * (panel_width + 12) + 12), panel_height + 91), '#f4f5f6')
    draw = ImageDraw.Draw(canvas)
    labels = [f'Before {dates[0][:10]}', f'Event {dates[1][:10]}', 'Radar darkening', 'Open-water candidates']
    for i, (label, panel) in enumerate(zip(labels, panels)):
        x = 12 + i * (panel_width + 12)
        draw.text((x, 14), label, fill='#18242f')
        canvas.paste(Image.fromarray(panel).resize((panel_width, panel_height), Image.Resampling.NEAREST), (x, 36))
    for x, code, text in [
        (12, 1, 'New darkening candidate'), (211, 2, 'Already dark'), (342, 255, 'Unknown'),
    ]:
        y = panel_height + 49
        draw.rectangle((x, y, x + 12, y + 12), fill=COLORS[code])
        draw.text((x + 18, y), text, fill='#18242f')
    draw.text((12, panel_height + 70), 'Same radar brightness scale. Candidate water requires confirmation.', fill='#263745')
    canvas.save(path)


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or 'T' not in value:
            raise ValueError
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use an ISO timestamp with time and UTC offset') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--event', type=Path, required=True)
    parser.add_argument('--before-time', type=timestamp, required=True)
    parser.add_argument('--event-time', type=timestamp, required=True)
    parser.add_argument('--track', type=int, required=True, help='Shared Sentinel-1 relative orbit from scene metadata')
    parser.add_argument('--polarization', choices=('VV', 'VH', 'HH', 'HV'), required=True)
    parser.add_argument('--dark-db', type=float, default=-17)
    parser.add_argument('--drop-db', type=float, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Optional observed or published flood polygons, WGS84 GeoJSON')
    parser.add_argument('--reference-aoi', type=Path, help='WGS84 GeoJSON with the reference mapping area')
    parser.add_argument('--reference-time', type=timestamp)
    args = parser.parse_args()
    reference_args = (args.reference, args.reference_aoi, args.reference_time)
    if any(reference_args) and not all(reference_args):
        parser.error('Reference scoring requires --reference, --reference-aoi, and --reference-time')
    if args.before_time >= args.event_time or not 1 <= args.track <= 175:
        parser.error('Expected an earlier baseline and a relative orbit from 1 to 175')
    if not np.isfinite(args.dark_db) or not np.isfinite(args.drop_db) or args.drop_db <= 0:
        parser.error('Thresholds must be finite and --drop-db must be positive')
    before, grid = read_backscatter(args.before)
    after, event_grid = read_backscatter(args.event)
    if grid != event_grid:
        parser.error('Inputs must share the same CRS, transform, and raster dimensions')
    classes, change_only = classify(before, after, args.dark_db, args.drop_db)
    valid = classes != 255
    crs, transform, width, height = grid
    dates = args.before_time.isoformat(), args.event_time.isoformat()
    provenance = {
        'before_time_utc': dates[0], 'event_time_utc': dates[1],
        'relative_orbit': args.track, 'polarization': args.polarization,
        'before_input': args.before.name, 'event_input': args.event.name,
    }
    stats = {**provenance, 'dark_db': args.dark_db, 'minimum_drop_db': args.drop_db,
             'darkening_pixels': int(change_only.sum()),
             'class_pixels': {label: int((classes == code).sum()) for code, label in CLASSES.items()}}
    if crs.is_projected and crs.linear_units in ('metre', 'meter'):
        stats['candidate_grid_area_km2'] = float((classes == 1).sum() * abs(transform.determinant) / 1e6)
    if args.reference:
        reference = geojson_mask(args.reference, grid)
        comparable = valid & geojson_mask(args.reference_aoi, grid)
        if not comparable.any() or not (reference & comparable).any():
            parser.error('Reference flood polygons and AOI must overlap the valid radar window')
        stats['reference_comparison'] = {
            'reference_time_utc': args.reference_time.isoformat(),
            'delta_hours': (args.event_time - args.reference_time).total_seconds() / 3600,
            'scored_pixels': int(comparable.sum()),
            'darkening_only': overlap(change_only, reference, comparable),
            'candidate_map': overlap(classes == 1, reference, comparable),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(driver='GTiff', width=width, height=height, count=1, crs=crs,
                    transform=transform, compress='deflate')
    with rasterio.open(args.output / 'water_candidates.tif', 'w', dtype='uint8', nodata=255, **metadata) as out:
        out.write(classes, 1)
        out.write_colormap(1, {0: (0, 0, 0), **COLORS})
        out.update_tags(**provenance, classes=json.dumps(CLASSES),
                        dark_db=args.dark_db, minimum_drop_db=args.drop_db)
    with rasterio.open(args.output / 'darkening.tif', 'w', dtype='uint8', nodata=255, **metadata) as out:
        out.write(np.where(valid, change_only, 255).astype('uint8'), 1)
        out.write_colormap(1, {0: (0, 0, 0), 1: COLORS[1], 255: COLORS[255]})
        out.update_tags(**provenance, minimum_drop_db=args.drop_db,
                        classes='0: no threshold crossing, 1: darkened, 255: unknown')
    for filename, band, description in (
        ('event_backscatter_db.tif', after, 'observed event radar backscatter in dB'),
        ('backscatter_change_db.tif', np.where(valid, after - before, np.nan), 'event minus before in dB'),
    ):
        with rasterio.open(args.output / filename, 'w', dtype='float32', nodata=np.nan, **metadata) as out:
            out.write(band, 1)
            out.set_band_description(1, description)
            out.update_tags(**provenance)
    preview(args.output / 'comparison.png', before, after, classes, change_only, dates)
    (args.output / 'summary.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
