"""Integration: the raster validator must enforce pair-to-manifest ownership."""
import csv
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from validate_data import validate


def make_tif(path, cloudy):
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = np.full((6, 128, 128), 10000, dtype=np.uint16)
    bands[4] = 2 if cloudy else 0
    bands[5] = 0
    with rasterio.open(path, 'w', driver='GTiff', height=128, width=128,
                       count=6, dtype='uint16', crs='EPSG:32645',
                       transform=from_origin(100000, 3000000, 30, 30)) as dst:
        dst.write(bands)


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_validator_rejects_cross_split_pair_even_when_rasters_align(tmp_path):
    source = 'data/cloudy/train/r00128_c00128.tif'
    target = 'data/reference/train/r00128_c00128.tif'
    make_tif(tmp_path / source, True)
    make_tif(tmp_path / target, False)
    write_csv(tmp_path / 'manifest.csv', ['path', 'category', 'split', 'row', 'col'],
              [{'path': source, 'category': 'cloudy', 'split': 'train', 'row': 128, 'col': 128}])
    write_csv(tmp_path / 'pairs.csv', ['input', 'target', 'split'],
              [{'input': source, 'target': target, 'split': 'train'}])
    validate(tmp_path)
    write_csv(tmp_path / 'pairs.csv', ['input', 'target', 'split'],
              [{'input': source, 'target': target, 'split': 'val'}])
    with pytest.raises(ValueError, match='belongs to'):
        validate(tmp_path)
