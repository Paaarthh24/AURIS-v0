"""Crack dataset loader with optional synthetic fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from auris.utils.config import CfgNode


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), Image.NEAREST)


class CrackDataset(Dataset):
    """Images + YOLO-normalized boxes + binary masks (all optional per sample)."""

    def __init__(self, cfg: CfgNode, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split
        self.size = int(cfg.model.input_size)
        self.mean = torch.tensor(list(cfg.data.mean), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(list(cfg.data.std), dtype=torch.float32).view(3, 1, 1)
        self.augment = split == "train"
        self.items: list[dict[str, Any]] = []
        if cfg.data.synthetic.enabled and not _has_real_split(cfg, split):
            self.items = _make_synthetic_index(cfg, split)
            self.synthetic = True
        else:
            self.synthetic = False
            self.items = _scan_real(cfg, split)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        if self.synthetic:
            image, boxes, mask = _render_synthetic(item, self.size, seed=item["seed"])
        else:
            image, boxes, mask = _load_real(item, self.size)

        if self.augment and self.cfg.data.augment.hflip and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            if boxes.numel():
                x1, y1, x2, y2 = boxes.unbind(1)
                boxes = torch.stack([self.size - x2, y1, self.size - x1, y2], dim=1)
        if self.augment and float(self.cfg.data.augment.color_jitter) > 0:
            jitter = float(self.cfg.data.augment.color_jitter)
            scale = 1.0 + (torch.rand(3, 1, 1) * 2 - 1) * jitter
            image = (image * scale).clamp(0, 1)

        image = (image - self.mean) / self.std
        labels = torch.zeros((boxes.shape[0],), dtype=torch.long)
        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "mask": mask,
            "id": item.get("id", str(index)),
        }


def collate_cracks(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    targets = [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch]
    return {
        "images": images,
        "masks": masks,
        "targets": targets,
        "ids": [b["id"] for b in batch],
    }


def build_dataloader(cfg: CfgNode, split: str) -> DataLoader:
    dataset = CrackDataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=split == "train",
        num_workers=int(cfg.data.num_workers),
        collate_fn=collate_cracks,
        drop_last=False,
        pin_memory=False,
    )


def _has_real_split(cfg: CfgNode, split: str) -> bool:
    root = Path(cfg.data.root)
    list_name = cfg.data.train_list if split == "train" else cfg.data.val_list
    return (root / list_name).is_file()


def _scan_real(cfg: CfgNode, split: str) -> list[dict[str, Any]]:
    root = Path(cfg.data.root)
    list_name = cfg.data.train_list if split == "train" else cfg.data.val_list
    names = [line.strip() for line in (root / list_name).read_text().splitlines() if line.strip()]
    items = []
    for name in names:
        stem = Path(name).stem
        image = root / cfg.data.image_dirname / name
        if not image.exists():
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = root / cfg.data.image_dirname / f"{stem}{ext}"
                if candidate.exists():
                    image = candidate
                    break
        items.append(
            {
                "id": stem,
                "image": image,
                "label": root / cfg.data.label_dirname / f"{stem}.txt",
                "mask": root / cfg.data.mask_dirname / f"{stem}.png",
            }
        )
    return items


def _load_real(item: dict[str, Any], size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = _to_tensor(_resize_rgb(Image.open(item["image"]), size))
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    label_path = Path(item["label"])
    if label_path.is_file():
        rows = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            x1 = (xc - w / 2) * size
            y1 = (yc - h / 2) * size
            x2 = (xc + w / 2) * size
            y2 = (yc + h / 2) * size
            rows.append([x1, y1, x2, y2])
        if rows:
            boxes = torch.tensor(rows, dtype=torch.float32)
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    mask_path = Path(item["mask"])
    if mask_path.is_file():
        m = _resize_mask(Image.open(mask_path).convert("L"), size)
        mask = (torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0) > 0.5).float().unsqueeze(0)
    return image, boxes, mask


def _make_synthetic_index(cfg: CfgNode, split: str) -> list[dict[str, Any]]:
    n = int(cfg.data.synthetic.train_size if split == "train" else cfg.data.synthetic.val_size)
    base = 0 if split == "train" else 10_000
    return [{"id": f"{split}_{i:04d}", "seed": int(cfg.seed) + base + i} for i in range(n)]


def _render_synthetic(
    item: dict[str, Any], size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    # Blue-green underwater-ish background with mild haze.
    bg = np.zeros((size, size, 3), dtype=np.float32)
    bg[..., 0] = rng.uniform(0.02, 0.12)
    bg[..., 1] = rng.uniform(0.18, 0.38)
    bg[..., 2] = rng.uniform(0.28, 0.55)
    noise = rng.normal(0, 0.03, bg.shape).astype(np.float32)
    image = np.clip(bg + noise, 0, 1)
    mask = np.zeros((size, size), dtype=np.float32)
    n_cracks = int(rng.randint(1, 4))
    boxes = []
    for _ in range(n_cracks):
        x0 = rng.randint(40, size - 40)
        y0 = rng.randint(40, size - 40)
        length = rng.randint(80, 220)
        angle = rng.uniform(-np.pi, np.pi)
        width = rng.randint(2, 6)
        xs, ys = [x0], [y0]
        for step in range(length):
            x = int(x0 + step * np.cos(angle) + rng.normal(0, 0.8))
            y = int(y0 + step * np.sin(angle) + rng.normal(0, 0.8))
            if 1 <= x < size - 1 and 1 <= y < size - 1:
                xs.append(x)
                ys.append(y)
                mask[max(0, y - width) : min(size, y + width + 1), max(0, x - width) : min(size, x + width + 1)] = 1.0
        if xs:
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            # darken crack on the image
            crack = mask > 0
            image[crack] = image[crack] * 0.25
    image_t = torch.from_numpy(image).permute(2, 0, 1)
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    box_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
    box_t = box_t.clamp(0, size - 1)
    return image_t, box_t, mask_t
