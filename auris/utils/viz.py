"""Overlay boxes and masks for qualitative checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def denormalize(image: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    mean_t = torch.tensor(mean, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=image.device, dtype=image.dtype).view(3, 1, 1)
    rgb = (image * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (rgb * 255).astype(np.uint8)


def draw_prediction(
    image: torch.Tensor,
    result: dict[str, torch.Tensor],
    mean: list[float],
    std: list[float],
    path: str | Path,
    gt_boxes: torch.Tensor | None = None,
    gt_mask: torch.Tensor | None = None,
) -> None:
    canvas = denormalize(image, mean, std)
    overlay = canvas.copy()
    if "mask" in result:
        mask = (result["mask"].detach().cpu().numpy() > 0.5).astype(np.uint8)
        overlay[mask == 1] = (0.45 * overlay[mask == 1] + 0.55 * np.array([255, 64, 64])).astype(np.uint8)
    if gt_mask is not None:
        gm = (gt_mask.detach().cpu().numpy() > 0.5)
        if gm.ndim == 3:
            gm = gm[0]
        overlay[gm] = (0.6 * overlay[gm] + 0.4 * np.array([64, 255, 64])).astype(np.uint8)
    img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(img)
    if gt_boxes is not None:
        for box in gt_boxes.detach().cpu().tolist():
            draw.rectangle(box, outline=(0, 220, 0), width=2)
    if "boxes" in result:
        for box, score in zip(result["boxes"].detach().cpu().tolist(), result["scores"].detach().cpu().tolist()):
            draw.rectangle(box, outline=(255, 48, 48), width=2)
            draw.text((box[0] + 2, box[1] + 2), f"{score:.2f}", fill=(255, 255, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
