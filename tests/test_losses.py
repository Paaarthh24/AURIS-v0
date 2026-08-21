import torch

from auris.losses import MultiTaskLoss, box_ciou, dice_loss
from auris.utils.config import load_config
from pathlib import Path

from auris.factory import build_model

CONFIG = Path(__file__).resolve().parents[1] / "configs/auris_v0.yaml"


def test_ciou_identical_boxes():
    box = torch.tensor([[10.0, 10.0, 40.0, 50.0]])
    assert torch.isclose(box_ciou(box, box), torch.tensor([1.0]), atol=1e-5)


def test_dice_perfect():
    logits = torch.full((1, 1, 8, 8), 20.0)
    target = torch.ones(1, 1, 8, 8)
    assert dice_loss(logits, target).item() < 1e-3


def test_joint_loss_backward():
    cfg = load_config(CONFIG)
    cfg.task = "joint"
    model = build_model(cfg)
    criterion = MultiTaskLoss(cfg)
    images = torch.randn(2, 3, 640, 640)
    masks = torch.zeros(2, 1, 640, 640)
    masks[:, :, 100:140, 80:400] = 1
    boxes = torch.tensor([[80.0, 100.0, 400.0, 140.0]])
    targets = [{"boxes": boxes, "labels": torch.zeros(1, dtype=torch.long)} for _ in range(2)]
    out = model(images, validate=False)
    losses = criterion(out, {"masks": masks, "targets": targets}, task="joint")
    losses["loss"].backward()
    assert "det" in losses and "seg" in losses
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(grads) > 0
