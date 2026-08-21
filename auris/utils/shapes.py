"""Tensor shape contracts for AURIS v0."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    shape: tuple[int | None, ...]

    def check(self, tensor: torch.Tensor) -> None:
        actual = tuple(tensor.shape)
        if len(actual) != len(self.shape):
            raise ShapeError(
                f"{self.name}: expected rank {len(self.shape)} {self.shape}, got {actual}"
            )
        for i, (exp, got) in enumerate(zip(self.shape, actual)):
            if exp is not None and exp != got:
                raise ShapeError(
                    f"{self.name}: dim {i} expected {exp}, got {got} (full {actual})"
                )


class ShapeError(ValueError):
    pass


def spatial_from_input(input_size: int, stride: int) -> int:
    if input_size % stride != 0:
        raise ShapeError(f"input_size {input_size} is not divisible by stride {stride}")
    return input_size // stride


def expected_backbone_shapes(
    batch: int, input_size: int, channels: list[int]
) -> dict[str, ShapeSpec]:
    strides = [4, 8, 16, 32]
    names = ["c2", "c3", "c4", "c5"]
    specs = {}
    for name, ch, stride in zip(names, channels, strides):
        hw = spatial_from_input(input_size, stride)
        specs[name] = ShapeSpec(name, (batch, ch, hw, hw))
    return specs


def expected_fpn_shapes(
    batch: int, input_size: int, out_channels: int
) -> dict[str, ShapeSpec]:
    specs = {}
    for name, stride in zip(["p2", "p3", "p4", "p5"], [4, 8, 16, 32]):
        hw = spatial_from_input(input_size, stride)
        specs[name] = ShapeSpec(name, (batch, out_channels, hw, hw))
    return specs


def validate_model_outputs(
    outputs: dict,
    batch: int,
    input_size: int,
    channels: list[int],
    fpn_channels: int,
    num_det_classes: int,
    det_enabled: bool,
    seg_enabled: bool,
) -> list[str]:
    """Validate a full forward dict. Returns a human-readable report."""
    lines: list[str] = []
    bb = expected_backbone_shapes(batch, input_size, channels)
    for name, spec in bb.items():
        spec.check(outputs["features"][name])
        lines.append(f"OK  features.{name}: {tuple(outputs['features'][name].shape)}")

    fpn = expected_fpn_shapes(batch, input_size, fpn_channels)
    for name, spec in fpn.items():
        spec.check(outputs["fpn"][name])
        lines.append(f"OK  fpn.{name}: {tuple(outputs['fpn'][name].shape)}")

    if det_enabled:
        det = outputs["detection"]
        for i, (level, stride) in enumerate(zip(["p2", "p3", "p4", "p5"], [4, 8, 16, 32])):
            hw = spatial_from_input(input_size, stride)
            ShapeSpec(f"det.cls.{level}", (batch, num_det_classes, hw, hw)).check(det["cls"][i])
            ShapeSpec(f"det.reg.{level}", (batch, 4, hw, hw)).check(det["reg"][i])
            ShapeSpec(f"det.obj.{level}", (batch, 1, hw, hw)).check(det["obj"][i])
            lines.append(
                f"OK  detection[{level}] cls={tuple(det['cls'][i].shape)} "
                f"reg={tuple(det['reg'][i].shape)} obj={tuple(det['obj'][i].shape)}"
            )
    elif outputs.get("detection") is not None:
        raise ShapeError("detection outputs present but detection is disabled")
    else:
        lines.append("OK  detection: disabled")

    if seg_enabled:
        logits = outputs["segmentation"]["logits"]
        ShapeSpec("seg.logits", (batch, 1, input_size, input_size)).check(logits)
        lines.append(f"OK  segmentation.logits: {tuple(logits.shape)}")
    elif outputs.get("segmentation") is not None:
        raise ShapeError("segmentation outputs present but segmentation is disabled")
    else:
        lines.append("OK  segmentation: disabled")

    return lines
