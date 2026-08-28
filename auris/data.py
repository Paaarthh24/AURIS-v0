"""Crack dataset loader: on-disk AURIS/YOLO layouts with synthetic fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from auris.utils.config import CfgNode

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), Image.NEAREST)


def split_list_name(cfg: CfgNode, split: str) -> str:
    if split == "train":
        return str(cfg.data.train_list)
    if split == "val":
        return str(cfg.data.val_list)
    if split == "test":
        return str(cfg.data.get("test_list", "test.txt"))
    raise ValueError(f"Unknown split: {split}")


class CrackDataset(Dataset):
    """Images + YOLO boxes (or YOLO-seg polygons) + binary masks."""

    def __init__(self, cfg: CfgNode, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split
        self.size = int(cfg.model.input_size)
        self.mean = torch.tensor(list(cfg.data.mean), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(list(cfg.data.std), dtype=torch.float32).view(3, 1, 1)
        self.augment = split == "train"
        self.items: list[dict[str, Any]] = []
        if _use_synthetic(cfg, split):
            self.items = _make_synthetic_index(cfg, split)
            self.synthetic = True
        else:
            self.synthetic = False
            self.items = _scan_dataset(cfg, split)
            if not self.items:
                raise FileNotFoundError(
                    f"No samples for split={split} under {cfg.data.root}. "
                    "Run: python scripts/prepare_dataset.py generate"
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        if self.synthetic:
            from auris.datasets.generate import render_underwater_crack

            rgb, mask_np, boxes_list = render_underwater_crack(
                self.size,
                seed=item["seed"],
                min_cracks=int(self.cfg.data.synthetic.get("min_cracks", 1)),
                max_cracks=int(self.cfg.data.synthetic.get("max_cracks", 3)),
            )
            image = torch.from_numpy(rgb).permute(2, 0, 1)
            mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
            boxes = (
                torch.tensor(boxes_list, dtype=torch.float32)
                if boxes_list
                else torch.zeros((0, 4), dtype=torch.float32)
            )
        else:
            image, boxes, mask = _load_sample(item, self.size)

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
    if split == "train":
        batch_size = int(cfg.train.batch_size)
        shuffle = True
    else:
        batch_size = int(cfg.eval.get("batch_size", cfg.train.batch_size))
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg.data.num_workers),
        collate_fn=collate_cracks,
        drop_last=False,
        pin_memory=False,
    )


def _use_synthetic(cfg: CfgNode, split: str) -> bool:
    synthetic = cfg.data.get("synthetic")
    enabled = bool(synthetic.get("enabled", False)) if synthetic is not None else False
    if not enabled:
        return False
    return not _has_split(cfg, split)


def _has_split(cfg: CfgNode, split: str) -> bool:
    root = Path(cfg.data.root)
    list_path = root / split_list_name(cfg, split)
    if list_path.is_file() and list_path.read_text(encoding="utf-8").strip():
        return True
    return _yolo_image_dir(root, split).is_dir()


def _yolo_image_dir(root: Path, split: str) -> Path:
    nested = root / "images" / split
    if nested.is_dir():
        return nested
    return root / split / "images"


def _scan_dataset(cfg: CfgNode, split: str) -> list[dict[str, Any]]:
    root = Path(cfg.data.root)
    list_path = root / split_list_name(cfg, split)
    if list_path.is_file() and list_path.read_text(encoding="utf-8").strip():
        return _scan_list(cfg, root, list_path)
    image_dir = _yolo_image_dir(root, split)
    label_dir = root / "labels" / split
    if not label_dir.is_dir():
        label_dir = root / split / "labels"
    mask_dir = root / str(cfg.data.get("mask_dirname", "masks")) / split
    if not mask_dir.is_dir():
        mask_dir = root / split / "masks"
    items = []
    if not image_dir.is_dir():
        return items
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        items.append(
            {
                "id": f"{split}_{image.stem}",
                "image": image,
                "label": label_dir / f"{image.stem}.txt",
                "mask": mask_dir / f"{image.stem}.png",
            }
        )
    return items


def _scan_list(cfg: CfgNode, root: Path, list_path: Path) -> list[dict[str, Any]]:
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    image_dirname = str(cfg.data.image_dirname)
    label_dirname = str(cfg.data.label_dirname)
    mask_dirname = str(cfg.data.mask_dirname)
    items = []
    for name in names:
        stem = Path(name).stem
        image = Path(name)
        if not image.is_absolute():
            image = root / image_dirname / name
            if not image.exists():
                image = root / name
        if not image.exists():
            for ext in IMAGE_EXTS:
                candidate = root / image_dirname / f"{stem}{ext}"
                if candidate.exists():
                    image = candidate
                    break
        items.append(
            {
                "id": stem,
                "image": image,
                "label": root / label_dirname / f"{stem}.txt",
                "mask": root / mask_dirname / f"{stem}.png",
            }
        )
    return items


def _parse_yolo_label(label_path: Path, width: int, height: int, size: int) -> tuple[torch.Tensor, np.ndarray | None]:
    if not label_path.is_file():
        return torch.zeros((0, 4), dtype=torch.float32), None
    boxes = []
    mask_img = None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        nums = list(map(float, parts[1:]))
        if len(nums) == 4:
            xc, yc, bw, bh = nums
            x1 = (xc - bw / 2) * size
            y1 = (yc - bh / 2) * size
            x2 = (xc + bw / 2) * size
            y2 = (yc + bh / 2) * size
            boxes.append([x1, y1, x2, y2])
        else:
            xs = [n * size for i, n in enumerate(nums) if i % 2 == 0]
            ys = [n * size for i, n in enumerate(nums) if i % 2 == 1]
            if len(xs) >= 3 and len(xs) == len(ys):
                if mask_img is None:
                    mask_img = Image.new("L", (size, size), 0)
                    draw = ImageDraw.Draw(mask_img)
                else:
                    draw = ImageDraw.Draw(mask_img)
                draw.polygon(list(zip(xs, ys)), fill=255)
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    box_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    mask_np = np.array(mask_img, dtype=np.uint8) if mask_img is not None else None
    return box_t, mask_np


def _load_sample(item: dict[str, Any], size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pil = Image.open(item["image"]).convert("RGB")
    orig_w, orig_h = pil.size
    image = _to_tensor(_resize_rgb(pil, size))
    boxes, poly_mask = _parse_yolo_label(Path(item["label"]), orig_w, orig_h, size)
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    mask_path = Path(item["mask"])
    if mask_path.is_file():
        m = _resize_mask(Image.open(mask_path).convert("L"), size)
        mask = (torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0) > 0.5).float().unsqueeze(0)
    elif poly_mask is not None:
        mask = torch.from_numpy((poly_mask > 0).astype(np.float32)).unsqueeze(0)
    if boxes.numel() == 0 and mask.any():
        from auris.datasets.generate import boxes_from_mask

        found = boxes_from_mask(mask[0].numpy().astype(np.uint8))
        if found:
            boxes = torch.tensor(found, dtype=torch.float32)
    return image, boxes, mask


def _make_synthetic_index(cfg: CfgNode, split: str) -> list[dict[str, Any]]:
    syn = cfg.data.synthetic
    if split == "train":
        n = int(syn.train_size)
        base = 0
    elif split == "val":
        n = int(syn.val_size)
        base = 10_000
    else:
        n = int(syn.get("test_size", syn.val_size))
        base = 20_000
    return [{"id": f"{split}_{i:04d}", "seed": int(cfg.seed) + base + i} for i in range(n)]
