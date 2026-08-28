"""Save training curves and detection PR/F1 plots under Results/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _xy(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in history:
        if key in row and row[key] is not None:
            xs.append(int(row["epoch"]))
            ys.append(float(row[key]))
    return xs, ys


def plot_history(history: list[dict[str, Any]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    def line(keys: list[str], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        drawn = False
        for key in keys:
            xs, ys = _xy(history, key)
            if xs:
                ax.plot(xs, ys, marker="o", label=key)
                drawn = True
        if not drawn:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)

    line(["train/loss", "val/loss"], "Loss", "loss", "loss.png")
    line(["train/det", "val/det", "val/det_ciou", "val/det_cls", "val/det_obj"], "Detection losses", "loss", "detection_loss.png")
    line(["val/det/precision"], "Precision", "precision", "precision.png")
    line(["val/det/recall"], "Recall", "recall", "recall.png")
    line(["val/det/f1", "val/score"], "F1 / score", "value", "f1.png")
    line(["lr"], "Learning rate", "lr", "lr.png")
    line(["val/det/ap"], "Average precision", "AP", "ap.png")


def plot_pr_bundle(curve: dict[str, Any], out_dir: str | Path, prefix: str = "val") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recall = curve.get("recall") or [0.0]
    precision = curve.get("precision") or [1.0]
    f1 = curve.get("f1") or [0.0]
    thresholds = curve.get("thresholds") or [1.0]
    ap = float(curve.get("ap") or 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{prefix} PR curve  (AP={ap:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_pr_curve.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1, lw=2, label="F1")
    ax.plot(thresholds, precision, lw=1.5, alpha=0.8, label="Precision")
    ax.plot(thresholds, recall, lw=1.5, alpha=0.8, label="Recall")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("metric")
    ax.set_title(f"{prefix} precision / recall / F1 vs confidence")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_f1_vs_confidence.png", dpi=140)
    plt.close(fig)
