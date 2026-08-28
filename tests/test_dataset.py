from pathlib import Path

from auris.data import CrackDataset, build_dataloader
from auris.datasets.convert import import_yolo_dataset, rasterize_yolo_label
from auris.datasets.generate import generate_dataset
from auris.datasets.local import index_detection_segmentation
from auris.utils.config import load_config
from PIL import Image, ImageDraw
import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / "configs/auris_v0.yaml"


def test_generate_and_load_splits(tmp_path):
    meta = generate_dataset(tmp_path, train=4, val=2, test=2, size=128, seed=0)
    assert meta["splits"] == {"train": 4, "val": 2, "test": 2}
    assert (tmp_path / "train.txt").is_file()
    cfg = load_config(CONFIG)
    cfg.data.root = str(tmp_path)
    cfg.data.synthetic.enabled = False
    cfg.data.root_fallbacks = []
    cfg.model.input_size = 128
    cfg.data.num_workers = 0
    cfg.train.batch_size = 2
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        ds = CrackDataset(cfg, split=split)
        assert len(ds) == n
        sample = ds[0]
        assert sample["image"].shape == (3, 128, 128)
        assert sample["mask"].shape == (1, 128, 128)
        assert sample["mask"].sum() > 0
        assert sample["boxes"].ndim == 2 and sample["boxes"].shape[1] == 4
        assert sample["boxes"].shape[0] >= 1
    batch = next(iter(build_dataloader(cfg, "test")))
    assert batch["images"].shape[0] == 2
    assert "targets" in batch


def test_yolo_seg_polygon_import(tmp_path):
    src = tmp_path / "yolo"
    img_dir = src / "images" / "train"
    lab_dir = src / "labels" / "train"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    image = Image.new("RGB", (64, 64), (20, 80, 120))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 50, 28), fill=(5, 10, 15))
    image.save(img_dir / "a.jpg")
    # triangle covering the crack band, YOLO-seg normalized
    lab_dir.joinpath("a.txt").write_text("0 0.15 0.30 0.85 0.30 0.85 0.45 0.15 0.45\n")
    out = tmp_path / "auris"
    counts = import_yolo_dataset(src, out, splits={"train": "train", "val": "val", "test": "test"})
    assert counts["train"] == 1
    mask = np.array(Image.open(out / "masks" / "train_a.png"))
    assert mask.max() > 0
    label = (out / "labels" / "train_a.txt").read_text().strip().split()
    assert len(label) == 5


def test_detection_segmentation_folder(tmp_path):
    det_img = tmp_path / "Detection" / "images"
    det_lab = tmp_path / "Detection" / "labels"
    seg_img = tmp_path / "Segmentation" / "images"
    seg_msk = tmp_path / "Segmentation" / "masks"
    for folder in (det_img, det_lab, seg_img, seg_msk):
        folder.mkdir(parents=True)
    Image.new("RGB", (64, 64), (10, 40, 80)).save(det_img / "joint.jpg")
    det_lab.joinpath("joint.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    Image.new("RGB", (64, 64), (10, 40, 80)).save(seg_img / "joint.jpg")
    mask = Image.new("L", (64, 64), 0)
    for x in range(22, 42):
        for y in range(28, 36):
            mask.putpixel((x, y), 255)
    mask.save(seg_msk / "joint.png")
    Image.new("RGB", (64, 64), (12, 50, 90)).save(seg_img / "seg_only.jpg")
    mask.save(seg_msk / "seg_only.png")

    splits = index_detection_segmentation(tmp_path, seed=0)
    total = sum(len(v) for v in splits.values())
    assert total == 2
    cfg = load_config(CONFIG)
    cfg.data.root = str(tmp_path)
    cfg.data.synthetic.enabled = False
    cfg.data.root_fallbacks = []
    cfg.model.input_size = 64
    cfg.data.num_workers = 0
    n = 0
    for split in ("train", "val", "test"):
        ds = CrackDataset(cfg, split=split)
        n += len(ds)
        if len(ds):
            sample = ds[0]
            assert sample["image"].shape == (3, 64, 64)
            assert sample["mask"].shape[0] == 1
    assert n == 2


def test_rasterize_bbox_label():
    from pathlib import Path as P
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = P(td) / "x.txt"
        path.write_text("0 0.5 0.5 0.4 0.2\n")
        mask, boxes = rasterize_yolo_label(path, 100, 100)
        assert len(boxes) == 1
        assert boxes[0][2] > boxes[0][0]
