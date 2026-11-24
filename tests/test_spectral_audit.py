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
