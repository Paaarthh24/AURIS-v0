"""On-disk underwater crack dataset generation (images, YOLO boxes, masks)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def value_noise(height: int, width: int, rng: np.random.RandomState, cells: int = 8) -> np.ndarray:
    grid = rng.rand(cells + 1, cells + 1).astype(np.float32)
    ys = np.linspace(0, cells, height, endpoint=False)
    xs = np.linspace(0, cells, width, endpoint=False)
    yi = np.floor(ys).astype(np.int32)
    xi = np.floor(xs).astype(np.int32)
    fy = (ys - yi)[:, None].astype(np.float32)
    fx = (xs - xi)[None, :].astype(np.float32)
    g00 = grid[yi[:, None], xi[None, :]]
    g10 = grid[yi[:, None] + 1, xi[None, :]]
    g01 = grid[yi[:, None], xi[None, :] + 1]
    g11 = grid[yi[:, None] + 1, xi[None, :] + 1]
    return (
        g00 * (1 - fy) * (1 - fx)
        + g10 * fy * (1 - fx)
        + g01 * (1 - fy) * fx
        + g11 * fy * fx
    )


def _polyline(rng: np.random.RandomState, size: int) -> list[tuple[int, int]]:
    x = rng.randint(24, size - 24)
    y = rng.randint(24, size - 24)
    angle = rng.uniform(-np.pi, np.pi)
    length = rng.randint(int(0.18 * size), int(0.55 * size))
    pts = [(int(x), int(y))]
    for _ in range(length):
        angle += rng.normal(0, 0.08)
        x += np.cos(angle) + rng.normal(0, 0.35)
        y += np.sin(angle) + rng.normal(0, 0.35)
        xi, yi = int(np.clip(x, 1, size - 2)), int(np.clip(y, 1, size - 2))
        if pts[-1] != (xi, yi):
            pts.append((xi, yi))
    return pts


def render_underwater_crack(
    size: int, seed: int, min_cracks: int = 1, max_cracks: int = 3
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Return RGB float image in [0,1], binary mask HxW, and xyxy boxes."""
    rng = np.random.RandomState(seed)
    concrete = 0.28 + 0.45 * value_noise(size, size, rng, cells=10)
    concrete += 0.08 * value_noise(size, size, rng, cells=32)
    concrete += rng.normal(0, 0.015, (size, size)).astype(np.float32)
    rgb = np.stack([concrete * 0.95, concrete * 0.92, concrete * 0.88], axis=-1)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    caustic = 0.04 * np.sin(xx / 18.0 + rng.uniform(0, 6)) * np.sin(yy / 27.0)
    rgb = rgb + caustic[..., None]

    # Underwater color cast + haze toward teal.
    cast = np.array([0.35, 0.78, 1.05], dtype=np.float32)
    haze = np.array([0.05, 0.22, 0.32], dtype=np.float32)
    depth = 0.25 + 0.35 * (yy / size)
    rgb = rgb * cast[None, None, :] * (1.0 - 0.35 * depth[..., None]) + haze * depth[..., None]
    rgb = np.clip(rgb, 0, 1)

    mask = np.zeros((size, size), dtype=np.uint8)
    n_cracks = int(rng.randint(min_cracks, max_cracks + 1))
    draw_mask = Image.fromarray(mask)
    painter = ImageDraw.Draw(draw_mask)
    boxes: list[list[float]] = []
    for _ in range(n_cracks):
        pts = _polyline(rng, size)
        width = int(rng.randint(2, 6))
        painter.line(pts, fill=255, width=width)
        for p in pts[:: max(len(pts) // 8, 1)]:
            r = width + int(rng.randint(0, 2))
            painter.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=255)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = width + 2
        boxes.append(
            [
                max(0, min(xs) - pad),
                max(0, min(ys) - pad),
                min(size - 1, max(xs) + pad),
                min(size - 1, max(ys) + pad),
            ]
        )
    mask = np.array(draw_mask, dtype=np.uint8)
    if rng.rand() < 0.7:
        mask = np.array(Image.fromarray(mask).filter(ImageFilter.MaxFilter(3)))

    crack = mask > 0
    rgb[crack] = rgb[crack] * rng.uniform(0.12, 0.28)
    rgb = np.clip(rgb + rng.normal(0, 0.01, rgb.shape).astype(np.float32), 0, 1)
    return rgb, (mask > 0).astype(np.uint8), boxes


def boxes_from_mask(mask: np.ndarray, min_area: int = 20) -> list[list[float]]:
    """Axis-aligned boxes from connected components (4-connected)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[list[float]] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        minx = maxx = x0
        miny = maxy = y0
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            minx, maxx = min(minx, x), max(maxx, x)
            miny, maxy = min(miny, y), max(maxy, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if area >= min_area:
            pad = 2
            boxes.append(
                [
                    max(0, minx - pad),
                    max(0, miny - pad),
                    min(w - 1, maxx + pad),
                    min(h - 1, maxy + pad),
                ]
            )
    return boxes


def yolo_line(box: list[float], width: int, height: int, cls: int = 0) -> str:
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def write_sample(
    out_root: Path,
    split: str,
    index: int,
    rgb: np.ndarray,
    mask: np.ndarray,
    boxes: list[list[float]] | None = None,
) -> str:
    name = f"{split}_{index:04d}.jpg"
    stem = Path(name).stem
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out_root / "images" / name, quality=90)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_root / "masks" / f"{stem}.png")
    h, w = mask.shape
    if not boxes:
        boxes = boxes_from_mask(mask)
    lines = [yolo_line(b, w, h) for b in boxes]
    (out_root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return name


def generate_dataset(
    out_root: str | Path,
    train: int = 48,
    val: int = 16,
    test: int = 16,
    size: int = 640,
    seed: int = 42,
    min_cracks: int = 1,
    max_cracks: int = 3,
) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    counts = {"train": train, "val": val, "test": test}
    written: dict[str, list[str]] = {}
    cursor = 0
    for split, n in counts.items():
        names = []
        for i in range(n):
            rgb, mask, boxes = render_underwater_crack(
                size=size, seed=seed + cursor, min_cracks=min_cracks, max_cracks=max_cracks
            )
            names.append(write_sample(out_root, split, i, rgb, mask, boxes))
            cursor += 1
        (out_root / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        written[split] = names

    meta = {
        "name": "auris_uwcrack",
        "description": "Synthetic underwater concrete crack dataset for AURIS v0 (RGB + YOLO boxes + binary masks).",
        "nc": 1,
        "names": ["crack"],
        "size": size,
        "seed": seed,
        "splits": {k: len(v) for k, v in written.items()},
        "layout": {
            "images": "images/*.jpg",
            "labels": "labels/*.txt  # YOLO class xc yc w h (normalized)",
            "masks": "masks/*.png    # binary 0/255",
            "lists": "train.txt, val.txt, test.txt",
        },
    }
    (out_root / "data.yaml").write_text(
        "\n".join(
            [
                "name: auris_uwcrack",
                "nc: 1",
                "names: [crack]",
                f"size: {size}",
                "train: train.txt",
                "val: val.txt",
                "test: test.txt",
                "images: images",
                "labels: labels",
                "masks: masks",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    readme = """# AURIS underwater crack dataset

Synthetic ROV-style concrete cracks with a blue-green underwater cast.

| Split | List | Contents |
| --- | --- | --- |
| train | `train.txt` | images + YOLO labels + masks |
| val | `val.txt` | same |
| test | `test.txt` | held-out evaluation |

Replace these files with real ROV captures using the same layout, or import a YOLO-seg dataset:

```bash
python scripts/prepare_dataset.py import-yolo --src path/to/yolo --out data/custom
```
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")
    return meta
