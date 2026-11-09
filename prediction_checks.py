"""Validate prepared Landsat patch coverage before reconstructing hidden pixels."""
import numpy as np

CLOUD_BITS = sum(1 << bit for bit in (1, 2, 3, 4))
SATURATION_BITS = sum(1 << bit for bit in (1, 2, 3, 4, 11))


def validate_prediction_patch(dn, qa, saturation):
    """Return a (H,W) cloud mask or reject unsupported input conditions."""
    dn, qa, saturation = map(np.asarray, (dn, qa, saturation))
    if dn.ndim != 3 or dn.shape[0] != 4 or qa.shape != dn.shape[1:] or saturation.shape != qa.shape:
        raise ValueError('Expected four optical bands and matching QA_PIXEL/QA_RADSAT grids')
    if dn.shape[1] % 8 or dn.shape[2] % 8:
        raise ValueError('Patch dimensions must be divisible by eight')
    if not all(np.issubdtype(x.dtype, np.integer) for x in (dn, qa, saturation)):
        raise ValueError('Expected integer Landsat DN and QA arrays')
    if np.any(dn == 0) or np.any(qa & 1) or np.any(saturation & SATURATION_BITS):
        raise ValueError('Patch contains invalid or saturated observations')
    mask = (qa & CLOUD_BITS) != 0
    if not mask.any():
        raise ValueError('No cloud/shadow pixels require reconstruction')
    if mask.all():
        raise ValueError('Entire patch is cloud/shadow: no clear optical context available')
    return mask.astype('float32')
