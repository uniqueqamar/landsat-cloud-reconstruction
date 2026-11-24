import numpy as np
import pytest
from prediction_checks import validate_prediction_patch, validate_reconstruction_output


def patch():
    return np.ones((4, 8, 8), dtype=np.uint16) * 20000, np.zeros((8, 8), dtype=np.uint16), np.zeros((8, 8), dtype=np.uint16)


def test_partial_cloud_mask():
    dn, qa, sat = patch(); qa[2, 3] = 1 << 3
    result = validate_prediction_patch(dn, qa, sat)
    assert result.shape == (8, 8) and result.dtype == np.float32
    assert result.sum() == 1 and result[2, 3] == 1


def test_all_cloud_rejected():
    dn, qa, sat = patch(); qa[:] = 1 << 3
    with pytest.raises(ValueError, match='Entire patch'):
        validate_prediction_patch(dn, qa, sat)


def test_clear_patch_rejected():
    dn, qa, sat = patch()
    with pytest.raises(ValueError, match='No cloud'):
        validate_prediction_patch(dn, qa, sat)


def test_saturation_rejected():
    dn, qa, sat = patch(); qa[0, 0] = 1 << 3; sat[2, 2] = 1 << 11
    with pytest.raises(ValueError, match='saturated'):
        validate_prediction_patch(dn, qa, sat)


def test_shape_rejected():
    dn, qa, sat = patch(); qa[0, 0] = 1 << 3
    with pytest.raises(ValueError, match='matching'):
        validate_prediction_patch(dn, qa[:-1], sat)


def test_invalid_dn_rejected():
    dn, qa, sat = patch(); qa[0, 0] = 1 << 3; dn[0, 0, 0] = 0
    with pytest.raises(ValueError, match='invalid'):
        validate_prediction_patch(dn, qa, sat)


def output_sample():
    observed = np.full((4, 8, 8), .1, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[0, 0] = 1
    predicted = observed.copy()
    predicted[:, 0, 0] = .2
    return predicted, observed, mask


def test_valid_output_preserves_clear_pixels():
    predicted, observed, mask = output_sample()
    assert validate_reconstruction_output(predicted, observed, mask) is predicted


def test_nonfinite_output_rejected():
    predicted, observed, mask = output_sample()
    predicted[:, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        validate_reconstruction_output(predicted, observed, mask)


def test_changed_clear_pixels_rejected():
    predicted, observed, mask = output_sample()
    predicted[:, 1, 1] = .2
    with pytest.raises(ValueError, match='clear pixels'):
        validate_reconstruction_output(predicted, observed, mask)


def test_output_shape_mismatch_rejected():
    predicted, observed, mask = output_sample()
    with pytest.raises(ValueError, match='shapes'):
        validate_reconstruction_output(predicted[:3], observed, mask)
