import numpy as np
import pytest

from expanded_shard_validation import validate_shard_rows


def shard(tmp_path, **overrides):
    data = dict(observed=np.ones((2, 4, 128, 128), dtype=np.uint16),
                target=np.ones((2, 4, 128, 128), dtype=np.uint16),
                qa_pixel=np.zeros((2, 128, 128), dtype=np.uint16))
    data.update(overrides)
    np.savez_compressed(tmp_path / 'shard.npz', **data)
    return [{'shard': 'shard.npz', 'index': '0'}, {'shard': 'shard.npz', 'index': '1'}]


def test_valid_shard(tmp_path):
    validate_shard_rows(tmp_path, shard(tmp_path))


def test_invalid_second_patch(tmp_path):
    target = np.ones((2, 4, 128, 128), dtype=np.uint16)
    target[1, 0, 0, 0] = 0
    with pytest.raises(AssertionError):
        validate_shard_rows(tmp_path, shard(tmp_path, target=target))


def test_invalid_index(tmp_path):
    rows = shard(tmp_path)
    rows[1]['index'] = '-1'
    with pytest.raises(AssertionError):
        validate_shard_rows(tmp_path, rows)


def test_wrong_qa_shape(tmp_path):
    rows = shard(tmp_path, qa_pixel=np.zeros((2, 64, 64), dtype=np.uint16))
    with pytest.raises(AssertionError):
        validate_shard_rows(tmp_path, rows)


def test_cloud_qa_invalid(tmp_path):
    qa = np.zeros((2, 128, 128), dtype=np.uint16)
    qa[0, 0, 0] = 32
    with pytest.raises(AssertionError):
        validate_shard_rows(tmp_path, shard(tmp_path, qa_pixel=qa))
