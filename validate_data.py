"""Check the actual delivered rasters, split separation, and optional pair alignment."""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio


def validate(root):
    rows = list(csv.DictReader((root / 'manifest.csv').open()))
    assert len({r['path'] for r in rows}) == len(rows)
    blocks = defaultdict(set)
    for row in rows:
        y, x = int(row['row']) // 128, int(row['col']) // 128
        assert y % 8 not in (0, 7) and x % 8 not in (0, 7)
        blocks[(y // 8, x // 8)].add(row['split'])
        with rasterio.open(root / row['path']) as src:
            assert src.shape == (128, 128) and src.count == 6
            assert src.res == (30., 30.) and str(src.crs) == 'EPSG:32645'
            assert np.all(src.read([1, 2, 3, 4]) != 0)
            qa, sat = src.read(5), src.read(6)
            assert not np.any(sat & sum(1 << b for b in (1, 2, 3, 4, 11)))
            assert not np.any(qa & 1)
            if row['category'] == 'clear':
                assert not np.any(qa & sum(1 << b for b in range(6)))
    assert all(len(splits) == 1 for splits in blocks.values())
    pair_count = 0
    if (root / 'pairs.csv').exists():
        for row in csv.DictReader((root / 'pairs.csv').open()):
            with rasterio.open(root / row['input']) as a, rasterio.open(root / row['target']) as b:
                assert (a.shape, a.crs, a.transform) == (b.shape, b.crs, b.transform)
                assert np.all(b.read([1, 2, 3, 4]) != 0)
                assert not np.any(b.read(5) & sum(1 << k for k in range(6)))
                assert not np.any(b.read(6) & sum(1 << k for k in (1, 2, 3, 4, 11)))
            pair_count += 1
    print(f'PASS: {len(rows)} patches; geographic split guards; {pair_count} aligned temporal pairs')


if __name__ == '__main__':
    validate(Path(__file__).resolve().parent)
