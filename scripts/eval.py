#!/usr/bin/env python3
"""Evaluate a checkpoint on val or test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auris.data import build_dataloader
from auris.engine import validate
from auris.factory import build_model
from auris.losses import MultiTaskLoss
from auris.utils.checkpoint import load_checkpoint
from auris.utils.config import load_config
from auris.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/auris_v0.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", choices=["det", "seg", "joint"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task:
        cfg.task = args.task
    seed_everything(int(cfg.seed), bool(cfg.deterministic))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg).to(device)
    ckpt_path = args.checkpoint or str(Path(cfg.train.output_dir) / "best.pt")
    if Path(ckpt_path).is_file():
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {ckpt_path}")
    else:
        print(f"No checkpoint at {ckpt_path}; evaluating randomly initialized weights.")
    loader = build_dataloader(cfg, args.split)
    logs = validate(
        model,
        loader,
        MultiTaskLoss(cfg),
        device,
        cfg,
        epoch=0,
        vis_dir=Path(cfg.train.output_dir) / f"vis_{args.split}",
        prefix=args.split,
    )
    print(json.dumps(logs, indent=2))
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_metrics.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
