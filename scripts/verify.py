#!/usr/bin/env python3
"""Verify tensor shapes, parameters, FLOPs, and a single training step.

Does not run full training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auris.engine import build_optimizer, build_scheduler
from auris.factory import build_model
from auris.losses import MultiTaskLoss
from auris.utils.config import load_config
from auris.utils.profile import count_macs, count_parameters, format_report, profile_latency
from auris.utils.seed import seed_everything
from auris.utils.shapes import validate_model_outputs


def _dummy_batch(batch: int, size: int, device: torch.device) -> dict:
    images = torch.randn(batch, 3, size, size, device=device)
    masks = torch.zeros(batch, 1, size, size, device=device)
    masks[:, :, 200:240, 100:500] = 1.0
    boxes = torch.tensor([[100.0, 200.0, 500.0, 240.0]], device=device)
    targets = [{"boxes": boxes, "labels": torch.zeros(1, dtype=torch.long, device=device)} for _ in range(batch)]
    return {"images": images, "masks": masks, "targets": targets}


def _task_cfg(path: Path, task: str):
    cfg = load_config(path)
    cfg.task = task
    if task == "det":
        cfg.model.detection.enabled = True
        cfg.model.segmentation.enabled = False
    elif task == "seg":
        cfg.model.detection.enabled = False
        cfg.model.segmentation.enabled = True
    else:
        cfg.model.detection.enabled = True
        cfg.model.segmentation.enabled = True
    return cfg


def verify_task(cfg, device: torch.device) -> dict:
    model = build_model(cfg).to(device)
    model.eval()
    batch = 2
    dummy = torch.randn(batch, 3, cfg.model.input_size, cfg.model.input_size, device=device)
    outputs = model(dummy, validate=True)
    shape_lines = validate_model_outputs(
        outputs,
        batch=batch,
        input_size=int(cfg.model.input_size),
        channels=list(cfg.model.backbone.channels),
        fpn_channels=int(cfg.model.fpn.out_channels),
        num_det_classes=int(cfg.model.detection.num_classes),
        det_enabled=model.det_enabled,
        seg_enabled=model.seg_enabled,
    )
    packed = _dummy_batch(batch, int(cfg.model.input_size), device)
    criterion = MultiTaskLoss(cfg)
    model.train()
    optimizer = build_optimizer(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(packed["images"], validate=False)
    losses = criterion(outputs, packed, task=model.task)
    losses["loss"].backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    return {
        "task": model.task,
        "shape_lines": shape_lines,
        "loss": float(losses["loss"].detach().cpu()),
        "loss_keys": sorted(k for k in losses if k != "det_pos"),
        "grad_ok": grad_ok,
        "params": count_parameters(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/auris_v0.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed), bool(cfg.deterministic))
    device = torch.device(args.device)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_shape_lines = []
    task_summaries = {}
    for task in ("det", "seg", "joint"):
        tcfg = _task_cfg(Path(args.config), task)
        summary = verify_task(tcfg, device)
        task_summaries[task] = {k: v for k, v in summary.items() if k != "shape_lines"}
        all_shape_lines.append(f"--- task={task} ---")
        all_shape_lines.extend(summary["shape_lines"])
        all_shape_lines.append(f"OK  one-step loss={summary['loss']:.6f} grad_ok={summary['grad_ok']}")

    joint_cfg = _task_cfg(Path(args.config), "joint")
    model = build_model(joint_cfg).to(device)
    dummy = torch.randn(1, 3, int(joint_cfg.model.input_size), int(joint_cfg.model.input_size), device=device)
    params = count_parameters(model)
    macs = count_macs(model, dummy)
    warmup = int(args.warmup if args.warmup is not None else cfg.profile.warmup)
    iters = int(args.iters if args.iters is not None else cfg.profile.iters)
    latency = profile_latency(model, dummy, warmup=warmup, iters=iters)
    report = format_report(params, macs, latency, extra_lines=all_shape_lines)

    (reports_dir / "auris_v0_profile.txt").write_text(report, encoding="utf-8")
    payload = {
        "params": params,
        "macs": macs,
        "latency": latency,
        "tasks": task_summaries,
        "shapes": all_shape_lines,
    }
    (reports_dir / "auris_v0_profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report)
    print(f"Wrote {reports_dir / 'auris_v0_profile.txt'}")


if __name__ == "__main__":
    main()
