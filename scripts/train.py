#!/usr/bin/env python3
"""Train AURIS v0. Prefer --verify-only until the shape/FLOP report is accepted."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auris.engine import fit
from auris.utils.config import load_config
from auris.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/auris_v0.yaml"))
    parser.add_argument("--task", choices=["det", "seg", "joint"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        import subprocess

        cmd = [sys.executable, str(ROOT / "scripts/verify.py"), "--config", args.config]
        if args.device:
            cmd += ["--device", args.device]
        raise SystemExit(subprocess.call(cmd))

    cfg = load_config(args.config)
    from auris.data import resolve_data_root

    print(f"AURIS data root: {resolve_data_root(cfg)}")
    if args.task:
        cfg.task = args.task
        if args.task == "det":
            cfg.model.segmentation.enabled = False
            cfg.model.detection.enabled = True
        elif args.task == "seg":
            cfg.model.detection.enabled = False
            cfg.model.segmentation.enabled = True
        else:
            cfg.model.detection.enabled = True
            cfg.model.segmentation.enabled = True
    seed_everything(int(cfg.seed), bool(cfg.deterministic))
    fit(cfg, max_epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
