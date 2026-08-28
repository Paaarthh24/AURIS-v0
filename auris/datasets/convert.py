"""Import YOLO detection / segmentation datasets into AURIS layout."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from auris.datasets.generate import boxes_from_mask, yolo_line

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.is_file():
            return path
    return None


def rasterize_yolo_label(label_path: Path, width: int, height: int) -> tuple[np.ndarray, list[list[float]]]:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    boxes: list[list[float]] = []
    if not label_path.is_file():
        return np.zeros((height, width), dtype=np.uint8), boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        nums = list(map(float, parts[1:]))
        if len(nums) == 4:
            xc, yc, bw, bh = nums
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            boxes.append([x1, y1, x2, y2])
            draw.rectangle([x1, y1, x2, y2], outline=255, width=max(2, int(min(width, height) * 0.004)))
        else:
            xs = [nums[i] * width for i in range(0, len(nums), 2)]
            ys = [nums[i] * height for i in range(1, len(nums), 2)]
            if len(xs) >= 3 and len(xs) == len(ys):
                poly = list(zip(xs, ys))
                draw.polygon(poly, fill=255)
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    arr = np.array(mask, dtype=np.uint8)
    if not boxes and arr.any():
        boxes = boxes_from_mask((arr > 0).astype(np.uint8))
    return (arr > 0).astype(np.uint8), boxes


def import_yolo_dataset(
    src: str | Path,
    out_root: str | Path,
    splits: dict[str, str] | None = None,
) -> dict[str, int]:
    """Convert a YOLO (det or seg) tree into AURIS images/labels/masks + split lists.

    Accepts either:
      src/images/{train,val,test} + src/labels/{train,val,test}
    or:
      src/{train,val,test}/images + labels
    """
    src = Path(src)
    out_root = Path(out_root)
    images_out = out_root / "images"
    labels_out = out_root / "labels"
    masks_out = out_root / "masks"
    for folder in (images_out, labels_out, masks_out):
        folder.mkdir(parents=True, exist_ok=True)

    splits = splits or {"train": "train", "val": "val", "test": "test"}
    counts: dict[str, int] = {}
    for split, alias in splits.items():
        image_dir = src / "images" / alias
        label_dir = src / "labels" / alias
        if not image_dir.is_dir():
            image_dir = src / alias / "images"
            label_dir = src / alias / "labels"
        if not image_dir.is_dir():
            counts[split] = 0
            (out_root / f"{split}.txt").write_text("", encoding="utf-8")
            continue
        names: list[str] = []
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            mask, boxes = rasterize_yolo_label(label_dir / f"{image_path.stem}.txt", w, h)
            dest_name = f"{split}_{image_path.stem}{image_path.suffix.lower()}"
            if dest_name.endswith(".jpeg"):
                dest_name = dest_name[:-5] + ".jpg"
            image.save(images_out / dest_name, quality=90)
            stem = Path(dest_name).stem
            Image.fromarray(mask * 255).save(masks_out / f"{stem}.png")
            if not boxes and mask.any():
                boxes = boxes_from_mask(mask)
            lines = [yolo_line(b, w, h) for b in boxes]
            (labels_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            names.append(dest_name)
        (out_root / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        counts[split] = len(names)
    (out_root / "data.yaml").write_text(
        "\n".join(
            [
                f"name: {out_root.name}",
                "nc: 1",
                "names: [crack]",
                "train: train.txt",
                "val: val.txt",
                "test: test.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return counts


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    urllib.request.urlretrieve(url, dest)
    return dest


def download_crack_seg(raw_dir: str | Path) -> Path:
    """Download Ultralytics crack-seg (road/wall cracks, YOLO-seg polygons)."""
    raw_dir = Path(raw_dir)
    zip_path = raw_dir / "crack-seg.zip"
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip"
    if not zip_path.is_file():
        download_file(url, zip_path)
    extract = raw_dir / "crack-seg"
    if not (extract / "images").is_dir() and not (extract / "train").is_dir():
        shutil.unpack_archive(zip_path, raw_dir)
        # zip may unwrap into crack-seg/ or datasets/crack-seg
        for candidate in (raw_dir / "crack-seg", raw_dir / "datasets" / "crack-seg"):
            if candidate.is_dir():
                extract = candidate
                break
        else:
            nested = [p for p in raw_dir.iterdir() if p.is_dir() and (p / "images").is_dir()]
            if nested:
                extract = nested[0]
    return extract
