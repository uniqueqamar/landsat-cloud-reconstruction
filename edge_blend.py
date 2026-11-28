"""Blend reconstructed pixels with spatial interpolation near the cloud boundary."""
import torch
from torch.nn import functional as F

from spatial_fill import spatial_fill


def edge_blend(image, mask, prediction, width, strength, *, interpolated=None):
    """Fade the blend to zero over the first width pixels inside the mask."""
    if width < 1 or not 0 <= strength <= 1:
        raise ValueError('Expected positive blend width and strength in [0, 1]')
    if interpolated is None:
        interpolated = spatial_fill(image, mask)
    interior = mask
    weight = torch.zeros_like(mask)
    for _ in range(width):
        # Four-neighbor erosion. Replication avoids treating the raster edge as clear.
        padded = F.pad(interior, (1, 1, 1, 1), mode='replicate')
        next_interior = interior * padded[..., :-2, 1:-1] * padded[..., 2:, 1:-1]
        next_interior = next_interior * padded[..., 1:-1, :-2] * padded[..., 1:-1, 2:]
        weight += mask - next_interior
        interior = next_interior
    weight *= strength / width
    return torch.where(mask.bool(), torch.lerp(prediction, interpolated, weight), image)
