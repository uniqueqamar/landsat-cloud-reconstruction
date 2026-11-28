"""Guard the mean-fill comparison against samples without observed pixels."""
import pytest
import torch
from torch import nn

from spectral_audit import audit


class IdentityModel(nn.Module):
    def forward(self, observed, mask):
        return observed


def test_all_cloudy_patch_rejected():
    observed = torch.ones((1, 4, 2, 2))
    mask = torch.ones((1, 1, 2, 2))
    target = torch.zeros_like(observed)
    with pytest.raises(ValueError, match='no clear pixels'):
        audit(IdentityModel(), [(observed, mask, target)], torch.device('cpu'))


def test_partly_clear_patch_scores():
    observed = torch.ones((1, 4, 2, 2)) * .2
    mask = torch.zeros((1, 1, 2, 2))
    mask[..., 0, 0] = 1
    target = observed.clone()
    scores = audit(IdentityModel(), [(observed, mask, target)], torch.device('cpu'))
    assert scores['model']['masked_mae'] == 0
    assert scores['mean_fill']['masked_mae'] == 0


def test_coverage_groups_share_the_same_mask_and_report_empty_groups():
    target = torch.full((3, 4, 2, 5), 0.25)
    mask = torch.zeros((3, 1, 2, 5))
    mask[0].view(-1)[:3] = 1
    mask[1].view(-1)[:5] = 1
    observed = target + 0.25 * mask

    scores = audit(IdentityModel(), [(observed, mask, target)], torch.device('cpu'),
                   coverage_breakdown=True)
    low, medium, high = scores['coverage_groups']
    assert [low['patch_count'], medium['patch_count'], high['patch_count']] == [1, 1, 0]
    assert high['metrics'] is None
    for method, expected_error in [('model', .25), ('mean_fill', 0),
                                   ('cloudy_input', .25), ('spatial_interpolation', 0)]:
        assert scores[method]['masked_pixel_count'] == 8
        assert low['metrics'][method]['masked_pixel_count'] == 3
        assert medium['metrics'][method]['masked_pixel_count'] == 5
        assert scores[method]['masked_mae'] == expected_error


def test_cloud_groups_partition_float_reconstruction_mask():
    from train import cloud_type_ids

    qa = torch.tensor([[0, 8 | 16, 16, 2]], dtype=torch.int32)
    categories = torch.from_numpy(cloud_type_ids(qa.numpy()))[None, None]
    mask = (categories > 0).float()
    target = torch.full((1, 4, 1, 4), .25)
    observed = target + .5 * mask
    scores = audit(IdentityModel(), [(observed, mask, target, categories)],
                   torch.device('cpu'), cloud_breakdown=True)
    groups = scores['cloud_types']
    assert [groups[k]['scored_pixels'] for k in ('cloud', 'cirrus', 'shadow', 'dilated_only')] == [1, 0, 1, 1]
    assert groups['cirrus']['metrics'] is None
    assert sum(entry['scored_pixels'] for entry in groups.values()) == scores['model']['masked_pixel_count']
    assert groups['shadow']['metrics']['model']['masked_mae'] == .5
