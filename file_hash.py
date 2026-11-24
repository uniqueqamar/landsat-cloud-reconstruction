"""Memory-bounded SHA-256 hashing for large raster data archives."""
import hashlib


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash a file incrementally without loading the entire archive into RAM."""
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()
