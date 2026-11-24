"""Validate expanded training patches without repeatedly decompressing each shard."""
from collections import defaultdict

import numpy as np


def validate_shard_rows(root, rows):
    """Check indexed patch contents, loading each shard once per validation pass.

    Index coordinates and split ownership are checked by the caller.
    """
    by_shard = defaultdict(list)
    for row in rows:
        by_shard[row['shard']].append(row)

    for name, shard_rows in by_shard.items():
        with np.load(root / name, allow_pickle=False) as archive:
            observed = archive['observed']
            target = archive['target']
            qa = archive['qa_pixel']
            assert observed.dtype == target.dtype == np.uint16
            assert observed.ndim == target.ndim == 4
            assert observed.shape == target.shape
            assert observed.shape[1:] == (4, 128, 128)
            assert qa.shape == (observed.shape[0], 128, 128)
            for row in shard_rows:
                i = int(row['index'])
                assert 0 <= i < observed.shape[0]
                assert np.all(observed[i] != 0) and np.all(target[i] != 0)
                assert not np.any(qa[i] & 33)
