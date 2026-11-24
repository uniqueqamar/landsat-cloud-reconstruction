"""Fast metadata-only leakage regression tests; no rasters or ML dependencies."""

import pytest

from pair_integrity import check_pair_assignments


SOURCE = 'data/cloudy/train/r00128_c00128.tif'
TARGET = 'data/reference/train/r00128_c00128.tif'


def manifest(category='cloudy', split='train'):
    return [{'path': SOURCE, 'category': category, 'split': split}]


def pairs(source=SOURCE, target=TARGET, split='train'):
    return [{'input': source, 'target': target, 'split': split}]


def test_valid_same_patch_temporal_pair():
    assert check_pair_assignments(manifest(), pairs()) is None


@pytest.mark.parametrize('bad_pair, message', [
    (pairs(split='test'), 'belongs to'),
    (pairs(source='data/cloudy/train/missing.tif'), 'absent from manifest'),
    (pairs(target='data/reference/test/r00128_c00128.tif'), 'same-patch reference'),
    (pairs(target='data/reference/train/different.tif'), 'same-patch reference'),
    (pairs(split='other'), 'unknown split'),
])
def test_invalid_pair_metadata_is_rejected(bad_pair, message):
    with pytest.raises(ValueError, match=message):
        check_pair_assignments(manifest(), bad_pair)


def test_clear_patch_cannot_be_used_as_cloudy_input():
    with pytest.raises(ValueError, match='not a cloudy patch'):
        check_pair_assignments(manifest(category='clear'), pairs())


def test_duplicate_pairs_cannot_overweight_evaluation():
    with pytest.raises(ValueError, match='duplicate input'):
        check_pair_assignments(manifest(), pairs() + pairs())


def test_duplicate_manifest_paths_rejected():
    with pytest.raises(ValueError, match='duplicate patch paths'):
        check_pair_assignments(manifest() + manifest(), pairs())
