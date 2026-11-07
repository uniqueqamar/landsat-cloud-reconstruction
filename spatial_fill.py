"""Deterministic spatial interpolation baseline for Landsat four-band patches."""
import torch
from torch.nn import functional as F


def spatial_fill(image, mask, iterations=128):
    """Diffuse clear pixels into cloud regions, preserving all observed pixels."""
    if image.ndim != 4 or image.shape[1] != 4 or mask.shape != (image.shape[0], 1, *image.shape[2:]):
        raise ValueError('Expected image (N,4,H,W) and mask (N,1,H,W)')
    if iterations < 1:
        raise ValueError('iterations must be positive')
    if not torch.isfinite(image).all() or not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Inputs must be finite with a binary mask')
    valid = 1 - mask
    if torch.any(valid.sum(dim=(2, 3)) == 0):
        raise ValueError('Cannot spatially fill a patch with no observed pixels')
    mean = (image * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True)
    current = image * valid + mean * mask
    kernel = image.new_tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]).reshape(1, 1, 3, 3) / 4
    for _ in range(iterations):
        padded = F.pad(current, (1, 1, 1, 1), mode='replicate')
        neighbors = F.conv2d(padded, kernel.expand(4, 1, 3, 3), groups=4)
        current = image * valid + neighbors * mask
    return current
