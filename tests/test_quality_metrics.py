import numpy as np
import pytest
from quality_metrics import MaskedQuality


def test_identical_four_band_reflectance():
    x = np.ones((4, 2, 2)) * .2
    q = MaskedQuality(); q.add(x, x, np.ones((1, 2, 2)))
    result = q.result()
    assert result['masked_mae'] == 0
    assert result['masked_rmse'] == 0
    assert result['masked_sam_degrees'] == pytest.approx(0)
    assert result['masked_ndvi_mae'] == 0
    assert result['masked_pixel_count'] == 4


def test_mask_excludes_unobserved_changes_and_streaming():
    target = np.ones((4, 1, 2)) * .2
    prediction = target.copy()
    prediction[:, 0, 0] += .1
    prediction[:, 0, 1] += 100
    q = MaskedQuality(); q.add(prediction, target, np.array([[1, 0]]))
    q.add(target, target, np.array([[1, 0]]))
    assert q.result()['masked_mae'] == pytest.approx(.05)
    assert q.result()['masked_pixel_count'] == 2


def test_ndvi_band_order():
    p = np.array([.1, .1, .2, .6]).reshape(4, 1, 1)
    r = np.array([.1, .1, .2, .2]).reshape(4, 1, 1)
    q = MaskedQuality(); q.add(p, r, np.ones((1, 1)))
    assert q.result()['masked_ndvi_mae'] == pytest.approx(.5)


def test_undefined_spectral_metrics_report_none():
    q = MaskedQuality(); q.add(np.zeros((4, 1, 1)), np.zeros((4, 1, 1)), np.ones((1, 1)))
    assert q.result()['masked_sam_degrees'] is None
    assert q.result()['masked_ndvi_mae'] is None


def test_empty_mask_rejected_on_result():
    q = MaskedQuality(); q.add(np.zeros((4, 1, 1)), np.zeros((4, 1, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match='No masked pixels'):
        q.result()


def test_shape_and_nonfinite_guard():
    q = MaskedQuality()
    with pytest.raises(ValueError):
        q.add(np.ones((3, 1, 1)), np.ones((3, 1, 1)), np.ones((1, 1)))
    p = np.ones((4, 1, 1)); p[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        q.add(p, np.ones((4, 1, 1)), np.ones((1, 1)))
