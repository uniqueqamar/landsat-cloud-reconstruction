"""Bounded caching of expanded training shards on memory-constrained machines."""
import csv

import numpy as np
import pytest
import torch

from train import TemporalPairs


def make_dataset(tmp_path, budget):
    (tmp_path / 'expanded_train').mkdir()
    with (tmp_path / 'pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'target', 'split'])
        writer.writeheader()
        writer.writerow({'input': 'unused.tif', 'target': 'unused-reference.tif', 'split': 'train'})
    with (tmp_path / 'expanded_train/index.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['shard', 'index', 'split'])
        writer.writeheader()
        for name in 'abc':
            path = f'expanded_train/{name}.npz'
            writer.writerow({'shard': path, 'index': 0, 'split': 'train'})
            observed = np.full((1, 4, 2, 2), 10000, dtype=np.uint16)
            target = np.full((1, 4, 2, 2), 20000, dtype=np.uint16)
            qa = np.zeros((1, 2, 2), dtype=np.uint16)
            qa[0, 0, 0] = 2
            np.savez_compressed(tmp_path / path, observed=observed, target=target, qa_pixel=qa)
    return TemporalPairs(tmp_path, 'train', expanded=True, shard_cache_limit_bytes=budget)


def test_lru_evicts_oldest_and_respects_byte_budget(tmp_path):
    ds = make_dataset(tmp_path, 144)  # Each shard: 32 + 32 + 8 = 72 bytes.
    ds[1]
    ds[2]
    assert list(ds.shard_cache) == ['expanded_train/a.npz', 'expanded_train/b.npz']
    ds[1]  # Refresh a; b becomes least recently used.
    ds[3]
    assert list(ds.shard_cache) == ['expanded_train/a.npz', 'expanded_train/c.npz']
    assert ds.shard_cache_bytes == 144


def test_oversized_shards_are_not_cached_and_values_remain_correct(tmp_path):
    ds = make_dataset(tmp_path, 71)
    observed, mask, target = ds[1]
    assert not ds.shard_cache
    assert ds.shard_cache_bytes == 0
    assert mask.shape == (1, 2, 2)
    assert mask.sum().item() == 1
    assert torch.allclose(target[:, 0, 0], torch.full((4,), 20000 * .0000275 - .2))
    assert torch.allclose(target[:, 1, 1], observed[:, 1, 1])


def test_zero_budget_disables_caching(tmp_path):
    ds = make_dataset(tmp_path, 0)
    ds[1]
    ds[2]
    assert ds.shard_cache_bytes == 0
    assert len(ds.shard_cache) == 0


def test_negative_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match='non-negative'):
        TemporalPairs(tmp_path, 'train', shard_cache_limit_bytes=-1)
