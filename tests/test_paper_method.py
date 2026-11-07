"""SQA and the evaluation protocol, checked against the authors' SQA/ code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from weperson.data.market1501 import Market1501
from weperson.metrics import evaluate_rank
from weperson.models.sqa import SQA, aug_samples, collect_params, configure_model, off_diagonal
from weperson.models.wabn import WABN2d

ROOT = Path(__file__).resolve().parents[1]


def test_wabn_statistics_follow_the_authors_formula():
    """Query and augmentations are mixed by w; running statistics move by 1 - alpha."""
    torch.manual_seed(0)
    bn = nn.BatchNorm2d(3)
    layer = WABN2d(bn, w=0.5, alpha=0.9).train()
    x = torch.randn(4, 3, 6, 6) * 2 + 1
    query, augmented = x[0], x[1:]
    mu = 0.5 * query.mean([1, 2]) + 0.5 * augmented.mean([0, 2, 3])
    var_query = ((query - mu[:, None, None]) ** 2).sum([1, 2]) / (36 - 1)
    var_aug = ((augmented - mu[:, None, None]) ** 2).sum([0, 2, 3]) / (3 * 36 - 1)
    expected_mean = 0.9 * bn.running_mean + 0.1 * mu
    expected_var = 0.9 * bn.running_var + 0.1 * (0.5 * var_query + 0.5 * var_aug)
    layer(x)
    assert torch.allclose(layer.running_mean, expected_mean, atol=1e-5)
    assert torch.allclose(layer.running_var, expected_var, atol=1e-5)


def test_wabn_keeps_statistics_at_evaluation():
    layer = WABN2d(nn.BatchNorm2d(3)).eval()
    before = layer.running_mean.clone()
    layer(torch.randn(2, 3, 4, 4) + 5.0)
    assert torch.equal(layer.running_mean, before)


def small_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())


def test_only_batch_norm_affine_parameters_adapt():
    """Abstract: SQA calibrates BatchNorm statistics and transformation parameters."""
    model = configure_model(small_model())
    params, names = collect_params(model)
    assert len(params) == 2
    assert all(name.endswith(("weight", "bias")) and name.startswith("1.") for name in names)
    assert not any(p.requires_grad for p in model[0].parameters())


def test_off_diagonal_drops_exactly_the_diagonal():
    matrix = torch.arange(9.0).view(3, 3)
    assert off_diagonal(matrix).tolist() == [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]


def test_query_is_kept_first_among_its_augmentations():
    x = torch.rand(1, 3, 32, 16)
    stacked = aug_samples(x, 8)
    assert stacked.shape == (9, 3, 32, 16)
    assert torch.equal(stacked[0], x[0])


def make_sqa() -> SQA:
    model = configure_model(small_model())
    params, _ = collect_params(model)
    return SQA(model, torch.optim.SGD(params, lr=1e-2))


def test_adaptation_changes_the_model_and_reset_restores_it():
    """Each query adapts from the source model, so one query cannot leak into the next."""
    torch.manual_seed(1)
    sqa = make_sqa()
    original = {k: v.clone() for k, v in sqa.model.state_dict().items()}
    sqa(torch.rand(1, 3, 32, 32) + 2.0)
    assert any(not torch.equal(v, original[k]) for k, v in sqa.model.state_dict().items())
    sqa.reset()
    assert all(torch.equal(v, original[k]) for k, v in sqa.model.state_dict().items())


def test_single_query_moves_statistics_towards_the_target_domain():
    """Abstract: a single target query narrows the gap to the unseen domain."""
    torch.manual_seed(2)
    sqa = make_sqa()
    layer = sqa.model[1]
    x = torch.rand(1, 3, 32, 32) * 4 + 3.0
    with torch.no_grad():
        target_mean = sqa.model[0](x).mean([0, 2, 3])
    before = (layer.running_mean - target_mean).norm().item()
    sqa(x)
    assert (layer.running_mean - target_mean).norm().item() < before


@pytest.mark.parametrize("key", ["lambd", "aug_n", "steps", "w", "alpha"])
def test_sqa_settings_match_the_paper(default_cfg, targets, key):
    assert default_cfg["sqa"][key] == targets["sqa_settings"][key]


def test_augmentations_match_the_paper():
    """Sec. V-B lists RandomResizedCrop; Sec. V-E says vertical flips distort ReID semantics."""
    source = (ROOT / "weperson" / "models" / "sqa.py").read_text(encoding="utf-8")
    assert "RandomResizedCrop" in source
    assert "RandomVerticalFlip" not in source


def rank(q_pid, q_cam, g_pids, g_cams, g_feats) -> dict:
    return evaluate_rank(
        torch.tensor([[1.0, 0.0]]), np.array([q_pid]), np.array([q_cam]), torch.tensor(g_feats),
        np.array(g_pids), np.array(g_cams), ranks=[1],
    )


def test_same_camera_matches_are_excluded():
    result = rank(1, 1, [1, 1, 2], [1, 2, 2], [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    assert result["Rank-1"] == 100.0


def test_junk_images_do_not_count_against_a_query(tmp_path):
    """Market-1501 protocol: id -1 images are junk and are ignored, unlike id 0 distractors."""
    (tmp_path / "bounding_box_test").mkdir()
    for name in ("0001_c2s1_000001_01.jpg", "-1_c2s1_000002_01.jpg", "0000_c2s1_000003_01.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / "bounding_box_test" / name)
    gallery = Market1501(tmp_path, "gallery")
    feats = {-1: [1.0, 0.0], 1: [0.8, 0.6], 0: [0.0, 1.0]}
    assert rank(1, 1, gallery.pids, gallery.camids, [feats[p] for p in gallery.pids])["Rank-1"] == 100.0


def test_checkpoints_are_not_selected_on_query_and_gallery():
    """Keeping the best epoch by test mAP makes the reported test mAP optimistic."""
    source = (ROOT / "weperson" / "train.py").read_text(encoding="utf-8")
    assert 'metrics["mAP"] > best["mAP"]' not in source
