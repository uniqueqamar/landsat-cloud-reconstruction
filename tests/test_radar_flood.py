import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from radar_flood import classify, read_backscatter


def test_darkening_outputs_keep_unknown_and_already_dark_separate():
    before = np.full((7, 7), -10., dtype='float32')
    after = before.copy()
    after[1:4, 1:4] = -20.
    before[6, 6], after[6, 6] = -18., -22.
    before[5, 0] = np.nan
    after[0, 6] = -22.

    classes, darkening = classify(before, after, dark_db=-17, drop_db=3)
    assert classes[2, 2] == 1
    assert classes[0, 6] == 0 and darkening[0, 6]
    assert classes[6, 6] == 2 and darkening[6, 6]
    assert classes[5, 0] == 255 and not darkening[5, 0]
    assert classes[6, 3] == 0 and not darkening[6, 3]


def test_linear_backscatter_uses_alpha_for_missing_observations(tmp_path):
    path = tmp_path / 'rtc.tif'
    with rasterio.open(
        path, 'w', driver='GTiff', count=2, dtype='float32', width=3, height=1,
        crs='EPSG:32722', transform=from_origin(0, 30, 30, 30),
    ) as target:
        target.colorinterp = (ColorInterp.gray, ColorInterp.alpha)
        target.write(np.array([[[.01, .01, 0]], [[255, 0, 255]]], dtype='float32'))

    values, _ = read_backscatter(path)
    np.testing.assert_allclose(values, [[-20, np.nan, np.nan]], atol=1e-5)
