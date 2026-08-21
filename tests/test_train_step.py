from pathlib import Path

from auris.data import CrackDataset, collate_cracks
from auris.engine import build_optimizer, build_scheduler, train_one_epoch
from auris.factory import build_model
from auris.losses import MultiTaskLoss
from auris.utils.config import load_config
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torch

CONFIG = Path(__file__).resolve().parents[1] / "configs/auris_v0.yaml"


def test_synthetic_dataset_shapes():
    cfg = load_config(CONFIG)
    cfg.data.num_workers = 0
    ds = CrackDataset(cfg, split="train")
    item = ds[0]
    assert item["image"].shape == (3, 640, 640)
    assert item["mask"].shape == (1, 640, 640)
    assert item["boxes"].ndim == 2 and item["boxes"].shape[1] == 4


def test_one_train_step_joint():
    cfg = load_config(CONFIG)
    cfg.task = "joint"
    cfg.data.synthetic.train_size = 4
    cfg.train.batch_size = 2
    cfg.data.num_workers = 0
    model = build_model(cfg)
    loader = DataLoader(CrackDataset(cfg, "train"), batch_size=2, collate_fn=collate_cracks)
    criterion = MultiTaskLoss(cfg)
    opt = build_optimizer(model, cfg)
    sch = build_scheduler(opt, cfg, steps_per_epoch=2)
    scaler = GradScaler(enabled=False)
    logs = train_one_epoch(
        model, loader, criterion, opt, sch, scaler, torch.device("cpu"), cfg, epoch=1
    )
    assert "train/loss" in logs
    assert logs["train/loss"] > 0
