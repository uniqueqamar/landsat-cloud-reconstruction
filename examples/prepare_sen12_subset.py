"""Prepare a bounded, scene-disjoint SEN12MS-CR subset from a pinned public mirror."""
import argparse
import json
from pathlib import Path
import re
import struct

import numpy as np
import pyarrow.parquet as pq
import requests
from s2cloudless import S2PixelCloudDetector


REVISION = '67cc2d43724a15167c050dd3060ede0e52d71e55'
BASE = f'https://huggingface.co/datasets/Hermanni/sen12mscr/resolve/{REVISION}/'
SCENES = (
    ('train', 'spring', 6, 10), ('train', 'fall', 11, 8), ('train', 'summer', 25, 12),
    ('validation', 'spring', 17, 11), ('test', 'spring', 31, 12), ('test', 'fall', 139, 12),
)
MAX_GROUP_BYTES = 225_000_000
CLOUD_SETTINGS = dict(threshold=.4, all_bands=True, average_over=4, dilation_size=2)


def range_response(url, start=None, stop=None, size=None):
    spec = f'{start}-{stop - 1}' if size is None else f'-{size}'
    response = requests.get(url, headers={'Range': 'bytes=' + spec}, stream=True, timeout=(15, 60))
    header = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
    if response.status_code != 206 or header is None:
        response.close()
        raise ValueError(f'Expected HTTP range support for {spec}')
    lower, upper, total = map(int, header.groups())
    expected = (start, stop - 1) if size is None else (total - size, total - 1)
    if expected != (lower, upper):
        response.close()
        raise ValueError(f'Source returned an unexpected byte range for {spec}')
    return response, lower, total


def fetch_group(cache, split, season, scene, index):
    destination = cache / f'{split}_{season}_scene_{scene}_subset.parquet'
    url = f'{BASE}{season}/scene_{scene}.parquet'
    range_size = 65536
    response, offset, total = range_response(url, size=range_size)
    with response:
        footer = response.raw.read(range_size + 1)
    if len(footer) != range_size or footer[-4:] != b'PAR1':
        raise ValueError('Expected a Parquet footer')
    length = struct.unpack('<I', footer[-8:-4])[0]
    if not 0 < length <= range_size - 8:
        raise ValueError('Parquet footer exceeds range budget')
    temporary = destination.with_suffix('.partial')
    output_partial = destination.with_suffix('.writing')
    try:
        temporary.write_bytes(b'PAR1' + footer[-8-length:])
        group = pq.read_metadata(temporary).row_group(index)
        chunks = []
        for column in range(group.num_columns):
            chunk = group.column(column)
            start = chunk.dictionary_page_offset or chunk.data_page_offset
            chunks.append((start, start + chunk.total_compressed_size))
        start = min(chunk[0] for chunk in chunks)
        stop = max(chunk[1] for chunk in chunks)
        if start < 4 or stop > total - length - 8 or stop - start > MAX_GROUP_BYTES:
            raise ValueError('Parquet group exceeds the download budget')
        # Sparse file. Only the requested group and the footer are materialized.
        with temporary.open('wb') as out:
            out.write(b'PAR1')
            out.truncate(total)
            out.seek(offset)
            out.write(footer)
            response, _, _ = range_response(f'{url}?row_group={index}', start, stop)
            with response:
                out.seek(start)
                for block in response.iter_content(1024 * 1024):
                    if out.tell() + len(block) > stop:
                        raise ValueError('Response exceeded the requested row group')
                    out.write(block)
                if out.tell() != stop:
                    raise ValueError('Incomplete parquet row group')
        pq.write_table(pq.ParquetFile(temporary).read_row_group(index), output_partial)
        output_partial.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
        output_partial.unlink(missing_ok=True)
    return destination


def prepare(row, detector):
    optical_shape = tuple(row['opt_shape'])
    cloudy = np.frombuffer(row['cloudy'], dtype='<i2').reshape(optical_shape)
    target = np.frombuffer(row['target'], dtype='<i2').reshape(optical_shape)
    if optical_shape != (256, 256, 13):
        raise ValueError(f'Unexpected optical shape {optical_shape}')
    valid = (cloudy[..., [1, 2, 3, 7]] > 0).all(axis=-1)
    if not valid.all() or not (target[..., [1, 2, 3, 7]] > 0).all():
        return None
    cloudy = np.clip(cloudy, 0, 10000).astype(np.float32) / 10000
    target = np.clip(target, 0, 10000).astype(np.float32) / 10000
    # Upstream SEN12MS-CR-TS loader's s2cloudless settings. Use cloudy input only.
    mask = detector.get_cloud_masks(cloudy[None], num_threads=2)[0].astype(bool)
    if not .02 <= mask.mean() <= .90:
        return None
    radar = np.frombuffer(row['sar'], dtype='<f4').reshape(row['sar_shape'])
    if radar.shape == (256, 256, 2):
        radar = np.moveaxis(radar, -1, 0)
    if radar.shape != (2, 256, 256):
        raise ValueError(f'Unexpected radar shape {radar.shape}')
    radar_valid = np.isfinite(radar).all(axis=0)
    if not radar_valid.any():
        return None
    floor = np.asarray([-25, -32.5], dtype=np.float32)[:, None, None]
    radar = np.clip((radar - floor) / -floor, 0, 1) - .5
    radar = np.where(radar_valid, radar, 0)
    return dict(
        observed=np.moveaxis(cloudy[..., [1, 2, 3, 7]], -1, 0),
        target=np.moveaxis(target[..., [1, 2, 3, 7]], -1, 0),
        mask=mask, radar=np.concatenate((radar, radar_valid[None]), axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use an empty output directory to avoid mixing data splits')
    args.parquet_cache.mkdir(parents=True, exist_ok=True)
    detector = S2PixelCloudDetector(**CLOUD_SETTINGS)
    manifest = []
    for split, season, scene, group in SCENES:
        cache = args.parquet_cache / f'{split}_{season}_scene_{scene}_subset.parquet'
        if not cache.exists():
            cache = fetch_group(args.parquet_cache, split, season, scene, group)
        directory = args.output / split
        directory.mkdir(parents=True, exist_ok=True)
        rows, ids = 0, []
        for batch in pq.ParquetFile(cache).iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            if row['season'] != season or int(row['scene']) != scene:
                raise ValueError(f'{cache}: unexpected scene or season')
            rows += 1
            prepared = prepare(row, detector)
            if prepared is None:
                continue
            identifier = f'{season}_{scene}_{row["patch"]}'
            if not re.fullmatch(r'[a-z]+_\d+_p\d+', identifier) or identifier in ids:
                raise ValueError(f'Unexpected or duplicate identifier {identifier}')
            ids.append(identifier)
            np.savez_compressed(directory / f'{identifier}.npz', **prepared)
        manifest.append(dict(split=split, season=season, scene=scene, row_group=group, rows=rows, patches=ids))
        print(split, season, scene, len(ids), '/', rows, flush=True)
    (args.output / 'manifest.json').write_text(json.dumps({
        'revision': REVISION, 'cloud_detector': 's2cloudless', 'settings': CLOUD_SETTINGS,
        'coverage_range': [.02, .90], 'scenes': manifest,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
