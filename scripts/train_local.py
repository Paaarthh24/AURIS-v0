#!/usr/bin/env python3
"""Train AURIS on the local Detection dataset with CUDA, then dump Results/.

Dataset (expected):
  /home/parth/Desktop/AURIS/auris-v0/Dataset/Detection/Images
  /home/parth/Desktop/AURIS/auris-v0/Dataset/Detection/Labels

Outputs:
  /home/parth/Desktop/AURIS/Results/
    plots/          loss, precision, recall, F1, PR curve, AP, lr
    detections/val  overlays + predictions.json + metrics
    detections/test overlays + predictions.json + metrics
    checkpoints     best.pt, last.pt (copied as best.pt / last.pt in Results/)
    history.json, config.yaml

Run from auris-v0 (laptop with RTX 5070):

  python scripts/train_local.py

  python scripts/train_local.py --epochs 80 --batch-size 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = Path("/home/parth/Desktop/AURIS/auris-v0/Dataset")
DEFAULT_RESULTS = Path("/home/parth/Desktop/AURIS/Results")


def _require_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        print("Running on CPU (slow).")
        return "cpu"
    if not torch.cuda.is_available():
        print(
            "CUDA is not available in this Python environment.\n"
            "Your RTX 5070 needs a CUDA build of PyTorch, for example:\n\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n\n"
            "Then check:\n"
            '  python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"\n'
        )
        raise SystemExit(1)
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"CUDA OK  device={name}  vram={mem:.1f} GiB  torch={torch.__version__}")
    torch.backends.cudnn.benchmark = True
    return "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/train_local.yaml"))
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset root containing Detection/")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Output folder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    import torch

    from auris.data import build_dataloader, resolve_data_root
    from auris.engine import export_detections, fit
    from auris.factory import build_model
    from auris.losses import MultiTaskLoss
    from auris.utils.checkpoint import load_checkpoint
    from auris.utils.config import load_config
    from auris.utils.seed import seed_everything

    device = _require_device(args.device)
    cfg = load_config(args.config)
    cfg.task = "det"
    cfg.model.detection.enabled = True
    cfg.model.segmentation.enabled = False
    cfg.data.root = args.data
    cfg.data.synthetic.enabled = False
    cfg.train.output_dir = args.results
    cfg.train.amp = device == "cuda"
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
        cfg.eval.batch_size = int(args.batch_size)
    if args.workers is not None:
        cfg.data.num_workers = int(args.workers)
    if args.resume:
        cfg.train.resume = args.resume

    data_root = Path(args.data)
    images = data_root / "Detection" / "Images"
    labels = data_root / "Detection" / "Labels"
    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(
            "Could not find Detection/Images and Detection/Labels.\n"
            f"  images: {images}  exists={images.is_dir()}\n"
            f"  labels: {labels}  exists={labels.is_dir()}\n"
            "Pass --data /home/parth/Desktop/AURIS/auris-v0/Dataset"
        )

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.seed), deterministic=False)

    print(f"Dataset root: {resolve_data_root(cfg)}")
    print(f"Images:       {images}")
    print(f"Labels:       {labels}")
    print(f"Results:      {results}")
    print(f"Epochs:       {cfg.train.epochs}  batch={cfg.train.batch_size}  AMP={cfg.train.amp}")

    fitted = fit(cfg, max_epochs=int(cfg.train.epochs), device=device)
    device_obj = torch.device(device)
    model = build_model(cfg).to(device_obj)
    best_path = results / "best.pt"
    ckpt_path = best_path if best_path.is_file() else results / "last.pt"
    if ckpt_path.is_file():
        ckpt = load_checkpoint(ckpt_path, map_location=device_obj)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {ckpt_path} for detection export")
    else:
        print("No checkpoint found; exporting the last in-memory weights.")
        if fitted.get("model") is not None:
            model = fitted["model"]

    criterion = MultiTaskLoss(cfg)
    summary = {"best": fitted.get("best"), "checkpoint": str(ckpt_path)}
    for split in ("val", "test"):
        try:
            loader = build_dataloader(cfg, split)
        except FileNotFoundError:
            print(f"Skipping {split}: no samples.")
            continue
        if len(loader.dataset) == 0:
            print(f"Skipping empty {split} split.")
            continue
        print(f"Exporting {split} detections ({len(loader.dataset)} images)...")
        logs = export_detections(model, loader, criterion, device_obj, cfg, results, split)
        summary[split] = logs
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in logs.items()}))

    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nDone. Inspect:\n  {results / 'plots'}\n  {results / 'detections'}\n  {results / 'best.pt'}")


if __name__ == "__main__":
    main()
