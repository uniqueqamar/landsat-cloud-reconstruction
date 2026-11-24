import pytest
import torch
from spatial_fill import spatial_fill


def test_preserves_clear_pixels_exactly():
    image = torch.rand(2, 4, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1
    out = spatial_fill(image, mask, 16)
    assert torch.equal(out * (1-mask), image * (1-mask))
    assert torch.isfinite(out).all()


def test_constant_scene_is_exact():
    image = torch.ones(1, 4, 8, 8) * .25
    mask = torch.ones(1, 1, 8, 8)
    mask[:, :, 0, :] = 0
    assert torch.allclose(spatial_fill(image, mask, 20), image, atol=1e-6)


def test_no_cloud_is_identity():
    image = torch.rand(1, 4, 4, 4)
    assert torch.equal(spatial_fill(image, torch.zeros(1, 1, 4, 4)), image)


def test_all_cloud_rejected():
    with pytest.raises(ValueError, match='no observed'):
        spatial_fill(torch.zeros(1, 4, 4, 4), torch.ones(1, 1, 4, 4))


def test_bad_mask_rejected():
    with pytest.raises(ValueError, match='binary'):
        spatial_fill(torch.zeros(1, 4, 4, 4), torch.ones(1, 1, 4, 4) * .5)


def test_gradient_flows_through_observed_pixels():
    image = torch.rand(1, 4, 5, 5, requires_grad=True)
    mask = torch.zeros(1, 1, 5, 5); mask[:, :, 2, 2] = 1
    spatial_fill(image, mask, 3).sum().backward()
    assert torch.isfinite(image.grad).all()
