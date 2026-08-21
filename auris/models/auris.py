"""AURIS v0 detector-segmentor."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from auris.models.backbone import ConvNeXtLiteBackbone
from auris.models.fpn import LightweightFPN
from auris.models.heads.detection import DetectionHead
from auris.models.heads.segmentation import SegmentationHead
from auris.models.hooks import assert_future_modules_disabled
from auris.utils.shapes import validate_model_outputs


class AURIS(nn.Module):
    """Lightweight multi-task model with independent det / seg / joint modes."""

    def __init__(self, cfg) -> None:
        super().__init__()
        assert_future_modules_disabled(cfg)
        self.cfg = cfg
        m = cfg.model
        self.input_size = int(m.input_size)
        self.task = str(cfg.task)
        self.det_enabled = bool(m.detection.enabled) and self.task in ("det", "joint")
        self.seg_enabled = bool(m.segmentation.enabled) and self.task in ("seg", "joint")
        if self.task == "det":
            self.seg_enabled = False
        if self.task == "seg":
            self.det_enabled = False
        if not self.det_enabled and not self.seg_enabled:
            raise ValueError("At least one of detection or segmentation must be enabled.")

        bb = m.backbone
        self.backbone = ConvNeXtLiteBackbone(
            in_channels=int(m.in_channels),
            channels=list(bb.channels),
            depths=list(bb.depths),
            kernel_size=int(bb.kernel_size),
            expansion_ratio=int(bb.expansion_ratio),
            drop_path_rate=float(bb.drop_path_rate),
            layer_scale_init=float(bb.layer_scale_init),
            stem_kernel=int(bb.stem_kernel),
            stem_stride=int(bb.stem_stride),
            attention_cfg=bb.attention,
            input_size=self.input_size,
        )
        self.fpn = LightweightFPN(
            in_channels=list(m.fpn.in_channels),
            out_channels=int(m.fpn.out_channels),
            input_size=self.input_size,
        )
        self.detection_head = None
        self.segmentation_head = None
        build_unused = bool(m.get("build_unused_heads", False))
        if self.det_enabled or (build_unused and m.detection.enabled):
            self.detection_head = DetectionHead(
                in_channels=int(m.detection.in_channels),
                num_classes=int(m.detection.num_classes),
                num_convs=int(m.detection.num_convs),
                strides=list(m.detection.strides),
                input_size=self.input_size,
            )
        if self.seg_enabled or (build_unused and m.segmentation.enabled):
            self.segmentation_head = SegmentationHead(
                in_channels=int(m.segmentation.in_channels),
                num_classes=int(m.segmentation.num_classes),
                input_size=self.input_size,
                upsample_mode=str(m.segmentation.upsample_mode),
            )

    def set_task(self, task: str) -> None:
        if task not in ("det", "seg", "joint"):
            raise ValueError(task)
        self.task = task
        self.det_enabled = task in ("det", "joint") and self.detection_head is not None
        self.seg_enabled = task in ("seg", "joint") and self.segmentation_head is not None
        if task == "det":
            self.seg_enabled = False
        if task == "seg":
            self.det_enabled = False

    def forward(
        self, images: torch.Tensor, validate: bool = True, return_features: bool = True
    ) -> dict[str, Any]:
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
        features = self.backbone(images, validate=validate)
        fpn = self.fpn(features, validate=validate)
        detection = None
        segmentation = None
        if self.det_enabled and self.detection_head is not None:
            detection = self.detection_head(fpn, validate=validate)
        if self.seg_enabled and self.segmentation_head is not None:
            segmentation = self.segmentation_head(fpn, validate=validate)
        outputs = {
            "features": features if return_features else None,
            "fpn": fpn if return_features else None,
            "detection": detection,
            "segmentation": segmentation,
        }
        if validate and return_features:
            validate_model_outputs(
                outputs,
                batch=images.shape[0],
                input_size=self.input_size,
                channels=self.backbone.channels,
                fpn_channels=self.fpn.out_channels,
                num_det_classes=int(self.cfg.model.detection.num_classes),
                det_enabled=self.det_enabled,
                seg_enabled=self.seg_enabled,
            )
        return outputs

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        conf_thresh: float | None = None,
        nms_iou: float | None = None,
        max_det: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        self.eval()
        cfg = self.cfg.model.detection
        conf_thresh = float(cfg.conf_thresh if conf_thresh is None else conf_thresh)
        nms_iou = float(cfg.nms_iou if nms_iou is None else nms_iou)
        max_det = int(cfg.max_det if max_det is None else max_det)
        outputs = self.forward(images, validate=False)
        batch = images.shape[0]
        results: list[dict[str, torch.Tensor]] = []
        decoded = None
        if self.det_enabled and outputs["detection"] is not None:
            decoded = decode_detections(
                outputs["detection"],
                strides=list(cfg.strides),
                conf_thresh=conf_thresh,
                nms_iou=nms_iou,
                max_det=max_det,
            )
        masks = None
        if self.seg_enabled and outputs["segmentation"] is not None:
            masks = torch.sigmoid(outputs["segmentation"]["logits"])
        for b in range(batch):
            item: dict[str, torch.Tensor] = {}
            if decoded is not None:
                item.update(decoded[b])
            else:
                item["boxes"] = images.new_zeros((0, 4))
                item["scores"] = images.new_zeros((0,))
                item["labels"] = images.new_zeros((0,), dtype=torch.long)
            if masks is not None:
                item["mask"] = masks[b, 0]
            results.append(item)
        return results


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float, max_det: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def decode_detections(
    detection: dict[str, list[torch.Tensor]],
    strides: list[int],
    conf_thresh: float,
    nms_iou: float,
    max_det: int,
) -> list[dict[str, torch.Tensor]]:
    cls_maps, reg_maps, obj_maps = detection["cls"], detection["reg"], detection["obj"]
    batch = cls_maps[0].shape[0]
    device = cls_maps[0].device
    results = []
    for b in range(batch):
        boxes_all, scores_all, labels_all = [], [], []
        for cls_map, reg_map, obj_map, stride in zip(cls_maps, reg_maps, obj_maps, strides):
            _, _, height, width = cls_map.shape
            ys, xs = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            cx = (xs + 0.5) * stride
            cy = (ys + 0.5) * stride
            ltrb = reg_map[b] * stride
            x1 = cx - ltrb[0]
            y1 = cy - ltrb[1]
            x2 = cx + ltrb[2]
            y2 = cy + ltrb[3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
            cls_prob = torch.sigmoid(cls_map[b]).reshape(cls_map.shape[1], -1)
            obj_prob = torch.sigmoid(obj_map[b]).reshape(-1)
            scores, labels = cls_prob.max(dim=0)
            scores = scores * obj_prob
            keep = scores >= conf_thresh
            boxes_all.append(boxes[keep])
            scores_all.append(scores[keep])
            labels_all.append(labels[keep])
        if boxes_all:
            boxes = torch.cat(boxes_all, dim=0)
            scores = torch.cat(scores_all, dim=0)
            labels = torch.cat(labels_all, dim=0)
        else:
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            labels = torch.zeros((0,), device=device, dtype=torch.long)
        keep = _nms(boxes, scores, nms_iou, max_det)
        results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
    return results
