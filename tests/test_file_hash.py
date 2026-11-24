import hashlib

import pytest

from file_hash import sha256_file


@pytest.mark.parametrize('payload', [b'', b'abc', bytes(range(256)) * 100])
def test_sha256_matches_hashlib(tmp_path, payload):
    path = tmp_path / 'shard.npz'
    path.write_bytes(payload)
    assert sha256_file(path, chunk_size=37) == hashlib.sha256(payload).hexdigest()


def test_hashing_reads_in_bounded_chunks(tmp_path):
    path = tmp_path / 'large-shard.npz'
    path.write_bytes(b'abc' * 5000)

    class GuardedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            assert 0 < size <= 128
            return self.stream.read(size)

    class GuardedPath:
        def open(self, mode):
            assert mode == 'rb'
            return GuardedStream(path.open(mode))

    assert sha256_file(GuardedPath(), chunk_size=128) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_invalid_chunk_size(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        sha256_file(tmp_path / 'missing', chunk_size=0)
