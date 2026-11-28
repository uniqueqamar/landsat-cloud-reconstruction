import numpy as np
import torch

from multimodal_gan import FusionGenerator, Sample, gradient_loss


def test_generator_inputs_never_use_covered_optical_or_reference_values():
    radar = np.ones((3, 5, 7), dtype=np.float32)
    observed = np.full((4, 5, 7), .3, dtype=np.float32)
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 1:5] = True
    sample = Sample('test', observed, observed.copy(), mask, radar)
    first, mean = sample.inputs(guided=True)
    sample.observed[:, mask] = .9
    sample.target[:] = .8
    second, second_mean = sample.inputs(guided=True)
    assert torch.equal(first, second)
    assert torch.equal(mean, second_mean)
    assert torch.count_nonzero(sample.inputs(guided=False)[0][:, 5:]) == 0
    assert FusionGenerator()(first, mean).shape == (1, 4, 5, 7)


def test_gradient_loss_scores_only_edges_between_two_masked_pixels():
    predicted = torch.tensor([.2, .3, .95]).view(1, 1, 1, 3).expand(1, 4, 1, 3)
    target = torch.zeros_like(predicted)
    mask = torch.tensor([True, True, False]).view(1, 1, 1, 3)
    torch.testing.assert_close(gradient_loss(predicted, target, mask), torch.tensor(.1))
