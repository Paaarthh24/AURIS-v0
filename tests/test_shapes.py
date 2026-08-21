from pathlib import Path

import pytest
import torch

from auris.factory import build_model
from auris.utils.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs/auris_v0.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG)


def test_joint_shapes(cfg):
    cfg.task = "joint"
    model = build_model(cfg)
    x = torch.randn(2, 3, 640, 640)
    out = model(x, validate=True)
    assert out["features"]["c2"].shape == (2, 32, 160, 160)
    assert out["features"]["c3"].shape == (2, 64, 80, 80)
    assert out["features"]["c4"].shape == (2, 128, 40, 40)
    assert out["features"]["c5"].shape == (2, 256, 20, 20)
    assert out["fpn"]["p2"].shape == (2, 96, 160, 160)
    assert out["fpn"]["p5"].shape == (2, 96, 20, 20)
    assert out["detection"]["cls"][0].shape == (2, 1, 160, 160)
    assert out["detection"]["reg"][3].shape == (2, 4, 20, 20)
    assert out["segmentation"]["logits"].shape == (2, 1, 640, 640)


def test_det_only_disables_seg(cfg):
    cfg.task = "det"
    cfg.model.segmentation.enabled = False
    model = build_model(cfg)
    out = model(torch.randn(1, 3, 640, 640), validate=True)
    assert out["detection"] is not None
    assert out["segmentation"] is None
    assert model.segmentation_head is None


def test_seg_only_disables_det(cfg):
    cfg.task = "seg"
    cfg.model.detection.enabled = False
    model = build_model(cfg)
    out = model(torch.randn(1, 3, 640, 640), validate=True)
    assert out["segmentation"] is not None
    assert out["detection"] is None
    assert model.detection_head is None


def test_future_attention_refused(cfg):
    cfg.model.backbone.attention.enabled = True
    cfg.model.backbone.attention.type = "se"
    with pytest.raises(NotImplementedError):
        build_model(cfg)
