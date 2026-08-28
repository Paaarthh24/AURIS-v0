#!/usr/bin/env python3
"""Create or import an AURIS train/val/test crack dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auris.datasets.convert import download_crack_seg, import_yolo_dataset
from auris.datasets.generate import generate_dataset
from auris.datasets.local import (
    find_named_dir,
    DET_DIR_NAMES,
    SEG_DIR_NAMES,
    index_detection_segmentation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Write the AURIS underwater crack dataset to disk")
    gen.add_argument("--out", default=str(ROOT / "data/auris_uwcrack"))
    gen.add_argument("--train", type=int, default=48)
    gen.add_argument("--val", type=int, default=16)
    gen.add_argument("--test", type=int, default=16)
    gen.add_argument("--size", type=int, default=640)
    gen.add_argument("--seed", type=int, default=42)

    imp = sub.add_parser("import-yolo", help="Convert a YOLO det/seg tree into AURIS layout")
    imp.add_argument("--src", required=True)
    imp.add_argument("--out", required=True)

    inspect = sub.add_parser(
        "inspect",
        help="Discover Detection/Segmentation folders and print split counts (no copy)",
    )
    inspect.add_argument(
        "--src",
        default="/home/parth/Desktop/AURIS/auris-v0/Dataset",
    )

    dl = sub.add_parser(
        "download-crack-seg",
        help="Download Ultralytics crack-seg and convert it (road/wall cracks, ~92MB)",
    )
    dl.add_argument("--raw", default=str(ROOT / "data/raw"))
    dl.add_argument("--out", default=str(ROOT / "data/crack_seg"))

    args = parser.parse_args()
    if args.cmd == "generate":
        meta = generate_dataset(
            args.out,
            train=args.train,
            val=args.val,
            test=args.test,
            size=args.size,
            seed=args.seed,
        )
        print(json.dumps(meta, indent=2))
        print(f"Wrote {args.out}")
        return
    if args.cmd == "import-yolo":
        counts = import_yolo_dataset(args.src, args.out)
        print(json.dumps({"out": args.out, "counts": counts}, indent=2))
        return
    if args.cmd == "inspect":
        src = Path(args.src).expanduser()
        if not src.is_dir():
            raise SystemExit(
                f"Dataset folder not found: {src}\n"
                "On this machine the path is missing. Run inspect/train on the PC "
                "that has /home/parth/Desktop/AURIS/auris-v0/Dataset, or copy/symlink "
                "it to ./Dataset"
            )
        det = find_named_dir(src, DET_DIR_NAMES)
        seg = find_named_dir(src, SEG_DIR_NAMES)
        splits = index_detection_segmentation(src)
        print(
            json.dumps(
                {
                    "root": str(src),
                    "detection_dir": str(det) if det else None,
                    "segmentation_dir": str(seg) if seg else None,
                    "counts": {k: len(v) for k, v in splits.items()},
                    "sample": (splits["train"][0] if splits["train"] else None),
                },
                indent=2,
            )
        )
        return
    if args.cmd == "download-crack-seg":
        extracted = download_crack_seg(args.raw)
        counts = import_yolo_dataset(extracted, args.out)
        print(json.dumps({"src": str(extracted), "out": args.out, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
