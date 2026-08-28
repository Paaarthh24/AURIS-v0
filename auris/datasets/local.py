"""Discover Detection + Segmentation folders and build a joint AURIS index."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXTS = {".txt", ".xml"}
DET_DIR_NAMES = {
    "detection",
    "detect",
    "det",
    "yolo",
    "bbox",
    "boxes",
    "damage detection",
    "damage_detection",
    "object detection",
}
SEG_DIR_NAMES = {
    "segmentation",
    "segment",
    "seg",
    "masks",
    "mask",
    "damage segmentation",
    "damage_segmentation",
    "semantic segmentation",
}
IMAGE_DIR_HINTS = {"image", "images", "img", "imgs", "jpeg", "jpg", "photos"}
MASK_DIR_HINTS = {"mask", "masks", "gt", "gts", "groundtruth", "ground_truth", "seg", "segs"}
LABEL_DIR_HINTS = {"label", "labels", "yolo", "voc", "xml", "bbox", "boxes", "annotations"}
SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
MASK_STEM_SUFFIXES = ("_mask", "-mask", "_gt", "-gt", "_label", "_seg", "_segmentation")


def _norm_name(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def find_named_dir(root: Path, names: set[str]) -> Path | None:
    if not root.is_dir():
        return None
    exact = []
    fuzzy = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        n = _norm_name(child.name)
        if n in names:
            exact.append(child)
        elif any(token in n for token in names if " " in token or len(token) > 3):
            if any(key in n for key in names):
                fuzzy.append(child)
    if exact:
        return exact[0]
    return fuzzy[0] if fuzzy else None


def infer_split(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts[:-1]:
        mapped = SPLIT_NAMES.get(part.lower())
        if mapped:
            return mapped
    return None


def normalize_stem(path: Path) -> str:
    stem = path.stem
    lower = stem.lower()
    for suffix in MASK_STEM_SUFFIXES:
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _parent_hint(path: Path) -> str:
    return _norm_name(path.parent.name)


def collect_files(task_root: Path, kind: str) -> dict[str, Path]:
    """kind: image | label | mask"""
    found: dict[str, Path] = {}
    if not task_root.is_dir():
        return found
    for path in sorted(task_root.rglob("*")):
        if not path.is_file():
            continue
        parent = _parent_hint(path)
        ext = path.suffix.lower()
        if kind == "label":
            if ext not in LABEL_EXTS:
                continue
            if ext == ".txt" and parent in MASK_DIR_HINTS:
                continue
            key = normalize_stem(path)
            found[key] = path
            continue
        if kind == "mask":
            if ext not in IMAGE_EXTS:
                continue
            stem_l = path.stem.lower()
            hinted = parent in MASK_DIR_HINTS or any(stem_l.endswith(s) for s in MASK_STEM_SUFFIXES)
            if hinted:
                found[normalize_stem(path)] = path
            continue
        if kind == "image":
            if not _is_image_file(path):
                continue
            if parent in MASK_DIR_HINTS:
                continue
            if parent in LABEL_DIR_HINTS:
                continue
            found[normalize_stem(path)] = path
    return found


def parse_voc_boxes(xml_path: Path, keep_classes: list[str] | None) -> list[list[float]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    keep = {c.lower() for c in keep_classes} if keep_classes else None
    boxes: list[list[float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if keep and name not in keep and "crack" not in name:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        x1 = float(box.findtext("xmin", "0"))
        y1 = float(box.findtext("ymin", "0"))
        x2 = float(box.findtext("xmax", "0"))
        y2 = float(box.findtext("ymax", "0"))
        boxes.append([x1, y1, x2, y2])
    return boxes


def assign_split(stem: str, hinted: str | None, seed: int = 42) -> str:
    if hinted:
        return hinted
    digest = hashlib.md5(f"{seed}:{stem}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def looks_like_task_dataset(root: Path) -> bool:
    if not root.is_dir():
        return False
    if find_named_dir(root, DET_DIR_NAMES) or find_named_dir(root, SEG_DIR_NAMES):
        return True
    if (root / "train.txt").is_file() or (root / "images").is_dir():
        return True
    return bool(collect_files(root, "image"))


def index_detection_segmentation(
    root: str | Path,
    keep_classes: list[str] | None = None,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Merge Detection labels and Segmentation masks by filename stem."""
    root = Path(root).expanduser().resolve()
    det_named = find_named_dir(root, DET_DIR_NAMES)
    seg_named = find_named_dir(root, SEG_DIR_NAMES)
    det_root = det_named or root
    seg_root = seg_named or root

    det_images = collect_files(det_root, "image")
    det_labels = collect_files(det_root, "label")
    seg_images = collect_files(seg_root, "image")
    seg_masks = collect_files(seg_root, "mask")

    stems = sorted(set(det_images) | set(seg_images) | set(det_labels) | set(seg_masks))
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for stem in stems:
        image = det_images.get(stem) or seg_images.get(stem)
        if image is None:
            continue
        label = det_labels.get(stem)
        mask = seg_masks.get(stem)
        hinted = infer_split(image, root)
        if hinted is None and label is not None:
            hinted = infer_split(label, root)
        if hinted is None and mask is not None:
            hinted = infer_split(mask, root)
        split = assign_split(stem, hinted, seed=seed)
        splits[split].append(
            {
                "id": stem,
                "image": str(image),
                "label": str(label) if label else "",
                "mask": str(mask) if mask else "",
            }
        )
    return splits


def write_split_lists(root: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    """Write train/val/test lists with image|label|mask absolute paths."""
    for split, items in splits.items():
        lines = []
        for item in items:
            lines.append("|".join([item["image"], item.get("label") or "", item.get("mask") or ""]))
        (root / f"{split}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
