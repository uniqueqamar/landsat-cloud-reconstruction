import pytest
import torch
from compare_spatial import compare


def test_comparison_on_constant_scene():
    x = torch.ones(1, 4, 6, 6) * .3
    m = torch.zeros(1, 1, 6, 6); m[:, :, 2:4, 2:4] = 1
    r = compare([(x, m, x)], iterations=4)
    assert r['spatial_masked_mae'] == pytest.approx(0, abs=1e-6)
    assert r['mean_fill_masked_mae'] == pytest.approx(0, abs=1e-6)
    assert r['masked_band_pixels'] == 16


def test_comparison_no_mask_rejected():
    x = torch.ones(1, 4, 4, 4)
    with pytest.raises(ValueError, match='No masked pixels'):
        compare([(x, torch.zeros(1, 1, 4, 4), x)])
