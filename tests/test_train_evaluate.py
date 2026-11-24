"""Regression coverage for empty cloud masks during validation."""
import pytest
import torch

from train import evaluate


class IdentityModel(torch.nn.Module):
    def forward(self, observed, mask):
        return observed


def sample(mask_value):
    return (
        torch.ones((1, 4, 2, 2)),
        torch.full((1, 1, 2, 2), float(mask_value)),
        torch.zeros((1, 4, 2, 2)),
    )


def test_all_clear_validation_has_actionable_error():
    with pytest.raises(ValueError, match="No masked pixels in val split"):
        evaluate(IdentityModel(), [sample(0)], torch.device("cpu"), "temporal")


def test_clear_samples_do_not_change_masked_mae():
    result = evaluate(IdentityModel(), [sample(0), sample(1)],
                      torch.device("cpu"), "temporal")
    assert result["masked_mae"] == 1.0
    assert result["skipped_unmasked_samples"] == 1


def test_regular_masked_sample_keeps_original_metric():
    result = evaluate(IdentityModel(), [sample(1)],
                      torch.device("cpu"), "synthetic")
    assert result["masked_mae"] == 1.0
    assert result["skipped_unmasked_samples"] == 0


def test_partly_clear_batch_counts_each_sample():
    observed = torch.ones((2, 4, 2, 2))
    mask = torch.cat([torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2))])
    target = torch.zeros_like(observed)
    result = evaluate(IdentityModel(), [(observed, mask, target)],
                      torch.device("cpu"), "temporal")
    assert result["masked_mae"] == 1.0
    assert result["skipped_unmasked_samples"] == 1
