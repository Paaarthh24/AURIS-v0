"""AURIS v0 — lightweight underwater crack detection and segmentation.

This repository implements **AURIS v0 core only**: a ConvNeXt-inspired CNN,
a 96-channel FPN, and independent anchor-free detection and binary
segmentation heads. Attention, knowledge distillation, depth estimation,
underwater enhancement, and adversarial domain adaptation are **not**
implemented; they are declared in YAML and raise `NotImplementedError` if
enabled so they can be added later without restructuring the backbone.

## Architecture

| Stage | Module | Default |
| --- | --- | --- |
| Input | RGB | `1×3×640×640` |
| Backbone | ConvNeXt-lite | channels `[32,64,128,256]`, depths `[2,2,4,2]`, 7×7 DWConv, LN, expand 4, GELU, residual |
| Features | C2–C5 | strides 4 / 8 / 16 / 32 → `160, 80, 40, 20` |
| Neck | Lightweight FPN | 96 channels, P2–P5 |
| Detection | Anchor-free (FCOS-style) | cls + ltrb + objectness per level |
| Segmentation | Binary fused FPN | logits `B×1×640×640` |

## Tasks

Configured by `task` in YAML (`det` | `seg` | `joint`):

- independent detection training
- independent segmentation training
- joint multi-task training

Losses: **CIoU + BCE** (detection), **BCE + Dice** (segmentation).
Optimizer: **AdamW** with AMP (CUDA) and **cosine LR + warm-up**.

## Setup

```bash
pip install -r requirements.txt
# CPU torch example:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Verify shapes, parameters, and FLOPs (required before full training)

```bash
python scripts/verify.py --config configs/auris_v0.yaml
```

Writes `reports/auris_v0_profile.txt` and `.json`. Do not start a full
training run until this report looks correct.

## Training (after verification)

```bash
python scripts/train.py --config configs/train_det.yaml --task det
python scripts/train.py --config configs/train_seg.yaml --task seg
python scripts/train.py --config configs/auris_v0.yaml --task joint
```

Synthetic underwater-like cracks are generated when no real split lists
exist. Real data layout:

```
data/root/
  images/*.png
  labels/*.txt    # YOLO: class xc yc w h (normalized)
  masks/*.png     # binary
  train.txt
  val.txt
```

## Adding later modules

Edit YAML only; register implementations in `auris/models/hooks.py`:

- `model.backbone.attention.type`: `se` | `eca` | `coord`
- `model.distillation`: feature / output KD
- `data.underwater_enhancement`
- `model.domain_adaptation` (adversarial transfer)
- `model.depth_estimation`

Keep `enabled: false` in v0.

## Tests

```bash
pytest -q
```
