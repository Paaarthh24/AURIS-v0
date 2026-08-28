"""Detection (P/R/F1/mAP@IoU) and segmentation (Dice/IoU) metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a.unbind(1)
    bx1, by1, bx2, by2 = b.unbind(1)
    inter_x1 = torch.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = torch.maximum(ay1[:, None], by1[None, :])
    inter_x2 = torch.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = torch.minimum(ay2[:, None], by2[None, :])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def ranked_matches(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> tuple[list[float], list[bool], int]:
    """Score-sorted TP/FP flags for a PR curve. Returns scores, is_tp, n_gt."""
    n_gt = int(gt_boxes.shape[0])
    if pred_boxes.numel() == 0:
        return [], [], n_gt
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    tps: list[bool] = []
    if n_gt:
        matched = torch.zeros(n_gt, dtype=torch.bool, device=gt_boxes.device)
        ious = box_iou(pred_boxes, gt_boxes)
        for i in range(pred_boxes.shape[0]):
            iou_i, j = ious[i].max(dim=0)
            hit = bool(iou_i >= iou_thresh and not matched[j])
            if hit:
                matched[j] = True
            tps.append(hit)
    else:
        tps = [False] * pred_boxes.shape[0]
    return pred_scores.detach().cpu().tolist(), tps, n_gt


def pr_curve(scores: list[float], tps: list[bool], n_gt: int) -> dict[str, list[float] | float]:
    if not scores or n_gt <= 0:
        return {
            "precision": [1.0],
            "recall": [0.0],
            "f1": [0.0],
            "thresholds": [1.0],
            "ap": 0.0,
        }
    import numpy as np

    scores_a = np.asarray(scores, dtype=np.float64)
    tps_a = np.asarray(tps, dtype=bool)
    order = np.argsort(-scores_a)
    tps_a = tps_a[order]
    scores_a = scores_a[order]
    cum_tp = np.cumsum(tps_a)
    cum_fp = np.cumsum(~tps_a)
    recall = cum_tp / float(n_gt)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # VOC-style interpolated AP
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "thresholds": scores_a.tolist(),
        "ap": ap,
    }


def match_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> tuple[int, int, int]:
    """Greedy score-sorted matching. Returns tp, fp, fn."""
    n_gt = gt_boxes.shape[0]
    if pred_boxes.numel() == 0:
        return 0, 0, n_gt
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    matched = torch.zeros(n_gt, dtype=torch.bool, device=gt_boxes.device)
    tp = fp = 0
    if n_gt:
        ious = box_iou(pred_boxes, gt_boxes)
        for i in range(pred_boxes.shape[0]):
            iou_i, j = ious[i].max(dim=0)
            if iou_i >= iou_thresh and not bool(matched[j]):
                tp += 1
                matched[j] = True
            else:
                fp += 1
    else:
        fp = pred_boxes.shape[0]
    fn = int((~matched).sum().item()) if n_gt else 0
    return tp, fp, fn


def binary_seg_stats(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> dict[str, float]:
    pred_bin = pred > thresh
    target_bin = target > 0.5
    inter = (pred_bin & target_bin).sum().item()
    pred_sum = pred_bin.sum().item()
    tgt_sum = target_bin.sum().item()
    union = pred_sum + tgt_sum - inter
    dice = (2 * inter + 1e-7) / (pred_sum + tgt_sum + 1e-7)
    iou = (inter + 1e-7) / (union + 1e-7)
    precision = (inter + 1e-7) / (pred_sum + 1e-7)
    recall = (inter + 1e-7) / (tgt_sum + 1e-7)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


@dataclass
class MetricMeter:
    det_tp: int = 0
    det_fp: int = 0
    det_fn: int = 0
    seg: dict[str, float] = field(default_factory=lambda: {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0})
    n_seg: int = 0
    n_det: int = 0

    def update_det(self, pred_boxes, pred_scores, gt_boxes, iou_thresh: float) -> None:
        tp, fp, fn = match_detections(pred_boxes, pred_scores, gt_boxes, iou_thresh)
        self.det_tp += tp
        self.det_fp += fp
        self.det_fn += fn
        self.n_det += 1

    def update_seg(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> None:
        stats = binary_seg_stats(pred_mask, gt_mask)
        for key, value in stats.items():
            self.seg[key] += value
        self.n_seg += 1

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        prec = self.det_tp / max(self.det_tp + self.det_fp, 1)
        rec = self.det_tp / max(self.det_tp + self.det_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-7)
        out["det/precision"] = prec
        out["det/recall"] = rec
        out["det/f1"] = f1
        if self.n_seg:
            for key, value in self.seg.items():
                out[f"seg/{key}"] = value / self.n_seg
        score_parts = [f1]
        if self.n_seg:
            score_parts.append(out["seg/dice"])
        out["score"] = sum(score_parts) / len(score_parts)
        return out
