"""Training / validation loop with AMP, cosine+warmup, checkpointing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from auris.data import build_dataloader
from auris.losses import MultiTaskLoss
from auris.metrics import MetricMeter
from auris.models.auris import AURIS
from auris.utils.checkpoint import load_checkpoint, save_checkpoint
from auris.utils.viz import draw_prediction


def build_optimizer(model: torch.nn.Module, cfg) -> AdamW:
    opt = cfg.train.optimizer
    if str(opt.type).lower() != "adamw":
        raise ValueError(f"AURIS v0 requires AdamW, got {opt.type}")
    return AdamW(
        model.parameters(),
        lr=float(opt.lr),
        weight_decay=float(opt.weight_decay),
        betas=tuple(opt.betas),
    )


def build_scheduler(optimizer: AdamW, cfg, steps_per_epoch: int) -> LambdaLR:
    sch = cfg.train.scheduler
    if str(sch.type).lower() != "cosine":
        raise ValueError(f"AURIS v0 requires cosine schedule, got {sch.type}")
    epochs = int(cfg.train.epochs)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup = int(sch.warmup_steps)
    warmup += int(sch.warmup_epochs) * steps_per_epoch
    base_lr = float(cfg.train.optimizer.lr)
    min_lr = float(sch.min_lr)
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return min_ratio + (1.0 - min_ratio) * float(step + 1) / float(max(warmup, 1))
        progress = (step - warmup) / float(max(total_steps - warmup, 1))
        import math

        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _amp_enabled(cfg, device: torch.device) -> bool:
    return bool(cfg.train.amp) and device.type == "cuda"


def train_one_epoch(
    model: AURIS,
    loader,
    criterion: MultiTaskLoss,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    cfg,
    epoch: int,
) -> dict[str, float]:
    model.train()
    task = model.task
    amp = _amp_enabled(cfg, device)
    running: dict[str, float] = {}
    n = 0
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["images"].to(device)
        batch["masks"] = batch["masks"].to(device)
        batch["targets"] = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in batch["targets"]
        ]
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            outputs = model(images, validate=False)
            losses = criterion(outputs, batch, task=task)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        if cfg.train.clip_grad_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.clip_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        n += 1
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu())
        if step % int(cfg.train.log_interval) == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu()), lr=optimizer.param_groups[0]["lr"])
    return {f"train/{k}": v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: AURIS,
    loader,
    criterion: MultiTaskLoss,
    device: torch.device,
    cfg,
    epoch: int,
    vis_dir: Path | None,
) -> dict[str, float]:
    model.eval()
    meter = MetricMeter()
    running: dict[str, float] = {}
    n = 0
    vis_count = 0
    max_vis = int(cfg.eval.max_vis)
    for batch in tqdm(loader, desc=f"val {epoch}", leave=False):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in batch["targets"]
        ]
        packed = {"masks": masks, "targets": targets}
        outputs = model(images, validate=False)
        losses = criterion(outputs, packed, task=model.task)
        n += 1
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu())
        results = model.predict(
            images,
            conf_thresh=float(cfg.eval.conf_thresh),
            nms_iou=float(cfg.eval.nms_iou),
        )
        for i, result in enumerate(results):
            if model.det_enabled:
                meter.update_det(
                    result["boxes"],
                    result["scores"],
                    targets[i]["boxes"],
                    float(cfg.eval.iou_thresh),
                )
            if model.seg_enabled:
                meter.update_seg(result["mask"], masks[i, 0])
            if vis_dir is not None and vis_count < max_vis and epoch % int(cfg.eval.vis_interval) == 0:
                draw_prediction(
                    images[i],
                    result,
                    list(cfg.data.mean),
                    list(cfg.data.std),
                    vis_dir / f"ep{epoch:03d}_{batch['ids'][i]}.png",
                    gt_boxes=targets[i]["boxes"],
                    gt_mask=masks[i, 0],
                )
                vis_count += 1
    logs = {f"val/{k}": v / max(n, 1) for k, v in running.items()}
    metrics = meter.compute()
    logs.update({f"val/{k}" if not k.startswith("val/") else k: v for k, v in metrics.items()})
    logs["val/score"] = metrics["score"]
    return logs


def fit(cfg, model: AURIS | None = None, max_epochs: int | None = None, device: str | None = None) -> dict[str, Any]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        model = AURIS(cfg)
    model.to(device_obj)
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")
    criterion = MultiTaskLoss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, max(len(train_loader), 1))
    scaler = GradScaler(enabled=_amp_enabled(cfg, device_obj))
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    start_epoch = 1
    best = -1e9
    if cfg.train.resume:
        ckpt = load_checkpoint(cfg.train.resume, map_location=device_obj)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best", best))

    epochs = int(max_epochs or cfg.train.epochs)
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs + 1):
        train_logs = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device_obj, cfg, epoch
        )
        val_logs = validate(
            model, val_loader, criterion, device_obj, cfg, epoch, vis_dir=out_dir / "vis"
        )
        logs = {**train_logs, **val_logs, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        history.append(logs)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in logs.items()}))
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "cfg": cfg.to_dict(),
        }
        if cfg.train.checkpoint.save_last:
            save_checkpoint(ckpt, out_dir / "last.pt")
        score = float(val_logs.get("val/score", 0.0))
        if cfg.train.checkpoint.save_best and score >= best:
            best = score
            ckpt["best"] = best
            save_checkpoint(ckpt, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "best": best, "output_dir": str(out_dir)}
