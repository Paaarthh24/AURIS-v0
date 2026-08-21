"""Detection (CIoU + BCE) and segmentation (BCE + Dice) losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def box_ciou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU between xyxy boxes. pred/target: [N, 4]."""
    px1, py1, px2, py2 = pred.unbind(1)
    tx1, ty1, tx2, ty2 = target.unbind(1)

    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2
    rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    enc_x1 = torch.minimum(px1, tx1)
    enc_y1 = torch.minimum(py1, ty1)
    enc_x2 = torch.maximum(px2, tx2)
    enc_y2 = torch.maximum(py2, ty2)
    c2 = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    v = (4 / (torch.pi**2)) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1).float()
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        dice = dice_loss(logits, targets, smooth=self.smooth)
        total = self.bce_weight * bce + self.dice_weight * dice
        return {"seg_bce": bce, "seg_dice": dice, "seg": total}


def _locations(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    return torch.stack([cx, cy], dim=-1).reshape(-1, 2)


def assign_fcos(
    boxes: torch.Tensor,
    locations: torch.Tensor,
    stride: int,
    regress_range: tuple[float, float],
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos_mask [L], ltrb [L,4]) for one image at one FPN level."""
    n_loc = locations.shape[0]
    if boxes.numel() == 0:
        return (
            torch.zeros(n_loc, dtype=torch.bool, device=locations.device),
            torch.zeros(n_loc, 4, device=locations.device),
        )
    cx, cy = locations[:, 0:1], locations[:, 1:2]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    l = cx - x1[None, :]
    t = cy - y1[None, :]
    r = x2[None, :] - cx
    b = y2[None, :] - cy
    ltrb = torch.stack([l, t, r, b], dim=-1)
    inside = ltrb.min(dim=-1).values > 0
    max_reg = ltrb.max(dim=-1).values
    in_range = (max_reg >= regress_range[0]) & (max_reg <= regress_range[1])

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    radius_px = radius * stride
    near_center = (
        (cx - center_x[None, :]).abs() <= radius_px
    ) & ((cy - center_y[None, :]).abs() <= radius_px)

    pos = inside & in_range & near_center
    areas = ((x2 - x1) * (y2 - y1)).clamp(min=1.0)
    area_map = areas[None, :].expand_as(pos).clone()
    area_map = area_map.masked_fill(~pos, 1e8)
    min_area, gt_idx = area_map.min(dim=1)
    pos_mask = min_area < 1e7
    matched = ltrb[torch.arange(n_loc, device=locations.device), gt_idx]
    matched = matched / stride
    matched = matched.masked_fill(~pos_mask[:, None], 0)
    return pos_mask, matched


class DetectionLoss(nn.Module):
    def __init__(
        self,
        strides: list[int],
        regress_ranges: list[list[float]],
        center_sampling_radius: float = 1.5,
        bce_cls_weight: float = 1.0,
        bce_obj_weight: float = 1.0,
        ciou_weight: float = 1.0,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.strides = list(strides)
        self.regress_ranges = [tuple(r) for r in regress_ranges]
        self.radius = center_sampling_radius
        self.bce_cls_weight = bce_cls_weight
        self.bce_obj_weight = bce_obj_weight
        self.ciou_weight = ciou_weight
        self.pos_weight = pos_weight

    def forward(
        self, detection: dict[str, list[torch.Tensor]], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        cls_maps, reg_maps, obj_maps = detection["cls"], detection["reg"], detection["obj"]
        device = cls_maps[0].device
        batch = cls_maps[0].shape[0]
        cls_losses, obj_losses, ciou_losses = [], [], []
        pos_count = 0

        for level, (cls_map, reg_map, obj_map, stride, rrange) in enumerate(
            zip(cls_maps, reg_maps, obj_maps, self.strides, self.regress_ranges)
        ):
            _, _, height, width = cls_map.shape
            locs = _locations(height, width, stride, device)
            for b in range(batch):
                boxes = targets[b]["boxes"].to(device)
                pos_mask, ltrb_tgt = assign_fcos(boxes, locs, stride, rrange, self.radius)
                pos_count += int(pos_mask.sum().item())

                cls_logit = cls_map[b].reshape(cls_map.shape[1], -1).transpose(0, 1)
                obj_logit = obj_map[b].reshape(-1)
                cls_tgt = pos_mask.float().unsqueeze(1).expand_as(cls_logit)
                obj_tgt = pos_mask.float()
                pw = cls_logit.new_tensor([self.pos_weight])
                cls_losses.append(
                    F.binary_cross_entropy_with_logits(cls_logit, cls_tgt, pos_weight=pw)
                )
                obj_losses.append(
                    F.binary_cross_entropy_with_logits(obj_logit, obj_tgt, pos_weight=pw)
                )

                if pos_mask.any():
                    pred_ltrb = reg_map[b].reshape(4, -1).transpose(0, 1)[pos_mask]
                    tgt_ltrb = ltrb_tgt[pos_mask]
                    cxcy = locs[pos_mask]
                    pred_boxes = torch.stack(
                        [
                            cxcy[:, 0] - pred_ltrb[:, 0] * stride,
                            cxcy[:, 1] - pred_ltrb[:, 1] * stride,
                            cxcy[:, 0] + pred_ltrb[:, 2] * stride,
                            cxcy[:, 1] + pred_ltrb[:, 3] * stride,
                        ],
                        dim=1,
                    )
                    tgt_boxes = torch.stack(
                        [
                            cxcy[:, 0] - tgt_ltrb[:, 0] * stride,
                            cxcy[:, 1] - tgt_ltrb[:, 1] * stride,
                            cxcy[:, 0] + tgt_ltrb[:, 2] * stride,
                            cxcy[:, 1] + tgt_ltrb[:, 3] * stride,
                        ],
                        dim=1,
                    )
                    ciou = box_ciou(pred_boxes, tgt_boxes)
                    ciou_losses.append(1.0 - ciou.mean())

        cls_loss = torch.stack(cls_losses).mean() if cls_losses else cls_maps[0].sum() * 0
        obj_loss = torch.stack(obj_losses).mean() if obj_losses else cls_maps[0].sum() * 0
        if ciou_losses:
            ciou_loss = torch.stack(ciou_losses).mean()
        else:
            ciou_loss = cls_maps[0].sum() * 0
        total = (
            self.bce_cls_weight * cls_loss
            + self.bce_obj_weight * obj_loss
            + self.ciou_weight * ciou_loss
        )
        return {
            "det_cls": cls_loss,
            "det_obj": obj_loss,
            "det_ciou": ciou_loss,
            "det": total,
            "det_pos": torch.tensor(float(pos_count), device=device),
        }


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        det_cfg = cfg.loss.detection
        seg_cfg = cfg.loss.segmentation
        self.det_weight = float(cfg.loss.task_weights.det)
        self.seg_weight = float(cfg.loss.task_weights.seg)
        self.det_loss = DetectionLoss(
            strides=list(cfg.model.detection.strides),
            regress_ranges=[list(r) for r in cfg.model.detection.regress_ranges],
            center_sampling_radius=float(cfg.model.detection.center_sampling_radius),
            bce_cls_weight=float(det_cfg.bce_cls_weight),
            bce_obj_weight=float(det_cfg.bce_obj_weight),
            ciou_weight=float(det_cfg.ciou_weight),
            pos_weight=float(det_cfg.pos_weight),
        )
        self.seg_loss = SegmentationLoss(
            bce_weight=float(seg_cfg.bce_weight),
            dice_weight=float(seg_cfg.dice_weight),
            smooth=float(seg_cfg.smooth),
        )

    def forward(self, outputs: dict[str, Any], batch: dict[str, Any], task: str) -> dict[str, torch.Tensor]:
        logs: dict[str, torch.Tensor] = {}
        total = None
        if task in ("det", "joint") and outputs.get("detection") is not None:
            det = self.det_loss(outputs["detection"], batch["targets"])
            logs.update(det)
            total = self.det_weight * det["det"]
        if task in ("seg", "joint") and outputs.get("segmentation") is not None:
            seg = self.seg_loss(outputs["segmentation"]["logits"], batch["masks"])
            logs.update(seg)
            total = seg["seg"] * self.seg_weight if total is None else total + self.seg_weight * seg["seg"]
        if total is None:
            raise RuntimeError(f"No losses computed for task={task}")
        logs["loss"] = total
        return logs
