import torch

from edge_blend import edge_blend


def test_blend_tapers_inside_boundary_and_preserves_clear_pixels():
    image = torch.full((1, 4, 3, 5), .25)
    mask = torch.ones((1, 1, 3, 5))
    mask[..., 0] = 0
    prediction = torch.ones_like(image)
    result = edge_blend(image, mask, prediction, width=2, strength=.5)
    expected = torch.tensor([.25, .625, .8125, 1., 1.])
    assert torch.equal(result, expected.expand_as(result))
