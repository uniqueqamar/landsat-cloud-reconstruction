"""Fast synthetic tests: no Landsat files or checkpoint downloads required."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_with_baseline


class FixedFill(torch.nn.Module):
    def forward(self, image, mask):
        return image * (1 - mask) + 2 * mask


def sample(cloud_value=10, target_value=2, mask_all=False):
    image = torch.ones(1, 4, 2, 2)
    image[:, :, -1, -1] = cloud_value
    target = torch.ones_like(image)
    target[:, :, -1, -1] = target_value
    mask = torch.zeros(1, 1, 2, 2)
    mask[:, :, -1, -1] = 1
    if mask_all:
        mask[:] = 1
    return image, mask, target


def test_paired_metrics_use_one_pass_and_keep_masked_weighting():
    class OnePass:
        passes = 0
        def __iter__(self):
            self.passes += 1
            yield sample(target_value=2)  # model MAE=0, mean fill MAE=1
            yield sample(target_value=1)  # model MAE=1, mean fill MAE=0
    loader = OnePass()
    result = evaluate_with_baseline(FixedFill(), loader, torch.device('cpu'), 'temporal', 'val', ['a', 'b'])
    assert loader.passes == 1
    assert result['masked_mae'] == pytest.approx(.5)
    assert result['mean_fill_masked_mae'] == pytest.approx(.5)
    assert result['occluded_input_masked_mae'] == pytest.approx(8.5)
    assert result['sample_count'] == 2
    assert (result['model_better_patch_count'], result['mean_fill_better_patch_count'], result['tied_patch_count']) == (1, 1, 0)
    assert result['median_patch_mae_delta'] == pytest.approx(0)
    assert result['per_patch'][0]['masked_fraction'] == pytest.approx(.25)
    assert [r['sample'] for r in result['per_patch']] == ['a', 'b']
    assert math.isfinite(result['masked_psnr_unit_range_db'])


def test_no_clear_pixels_rejected_instead_of_zero_fill():
    with pytest.raises(ValueError, match='both masked and clear'):
        evaluate_with_baseline(FixedFill(), [sample(mask_all=True)], torch.device('cpu'), 'synthetic')


def test_empty_dataset_rejected():
    with pytest.raises(ValueError, match='empty dataset'):
        evaluate_with_baseline(FixedFill(), [], torch.device('cpu'), 'synthetic')


def test_mismatched_target_shape_rejected():
    image, mask, target = sample()
    with pytest.raises(ValueError, match='target shape'):
        evaluate_with_baseline(FixedFill(), [(image, mask, target[:, :1])], torch.device('cpu'), 'temporal')


def test_nonbinary_mask_rejected():
    image, mask, target = sample()
    mask[:, :, -1, -1] = .5
    with pytest.raises(ValueError, match='finite binary'):
        evaluate_with_baseline(FixedFill(), [(image, mask, target)], torch.device('cpu'), 'temporal')


def test_wrong_prediction_shape_rejected():
    class WrongShape(torch.nn.Module):
        def forward(self, image, mask):
            return image[:, :1]
    with pytest.raises(ValueError, match='prediction shape'):
        evaluate_with_baseline(WrongShape(), [sample()], torch.device('cpu'), 'temporal')
