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

Writes `reports/auris_v0_profile.txt` and `.json`. Measured joint model at `1×3×640×640` (CPU):

- **2,056,171** parameters (backbone 1,899,872; FPN 87,552; det head 21,130; seg head 47,617)
- **4.233 GMACs** / **8.467 GFLOPs** (counting 2 FLOPs per MAC on conv+linear)
- Shape contracts for C2–C5, P2–P5, det maps, and 640×640 seg logits all pass for det / seg / joint

Do not start a full training run until this report looks correct.

## Local CUDA training (Detection/Images + Labels)

On the RTX 5070 laptop, from `auris-v0`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python scripts/train_local.py
```

Reads:

- `/home/parth/Desktop/AURIS/auris-v0/Dataset/Detection/Images`
- `/home/parth/Desktop/AURIS/auris-v0/Dataset/Detection/Labels`

Writes `/home/parth/Desktop/AURIS/Results/` (`plots/`, `detections/val`, `detections/test`, `best.pt`).

```bash
python scripts/train_local.py --epochs 80 --batch-size 8 --device cuda
```

## Dataset

Default training root is your local folder:

`/home/parth/Desktop/AURIS/auris-v0/Dataset`

AURIS looks for **Detection** and **Segmentation** subfolders (any similar name), joins samples by filename, and uses YOLO `.txt` or VOC `.xml` boxes plus PNG masks. If that path is missing, it falls back to `./Dataset` then `data/auris_uwcrack`.

```bash
python scripts/prepare_dataset.py inspect --src /home/parth/Desktop/AURIS/auris-v0/Dataset
python scripts/train.py --config configs/auris_v0.yaml --task joint
python scripts/eval.py --config configs/auris_v0.yaml --split test
```

Override with `AURIS_DATA_ROOT=/path/to/Dataset`.

Create or enlarge the bundled demo set:

```bash
python scripts/prepare_dataset.py generate --out data/auris_uwcrack --train 48 --val 16 --test 16
```

Import your own YOLO detection or YOLO-seg tree (polygons become masks):

```bash
python scripts/prepare_dataset.py import-yolo --src path/to/yolo --out data/custom
```

Optional public road/wall crack set (Ultralytics crack-seg, ~92MB) for extra experiments:

```bash
python scripts/prepare_dataset.py download-crack-seg --out data/crack_seg
```

Then point `data.root` in the YAML at that folder (`synthetic.enabled` is only a fallback when split lists are missing).

Held-out test:

```bash
python scripts/eval.py --config configs/auris_v0.yaml --split test --checkpoint runs/auris_v0/best.pt
```

Layout:

```
data/auris_uwcrack/
  images/*.jpg
  labels/*.txt    # YOLO: class xc yc w h (normalized)
  masks/*.png     # binary
  train.txt
  val.txt
  test.txt
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
