"""Independent binary segmentation head over the FPN pyramid."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from auris.models.fpn import ConvBNAct


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 96,
        num_classes: int = 1,
        input_size: int = 640,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("AURIS v0 segmentation is binary (num_classes=1).")
        self.input_size = input_size
        self.upsample_mode = upsample_mode
        self.level_names = ["p2", "p3", "p4", "p5"]
        self.project = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, kernel_size=1) for _ in self.level_names]
        )
        self.fuse = ConvBNAct(in_channels, in_channels, kernel=3)
        self.pred = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(
        self, fpn: dict[str, torch.Tensor], validate: bool = True
    ) -> dict[str, torch.Tensor]:
        target_hw = fpn["p2"].shape[-2:]
        fused = 0
        for name, proj in zip(self.level_names, self.project):
            feat = proj(fpn[name])
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(
                    feat, size=target_hw, mode=self.upsample_mode, align_corners=False
                )
            fused = fused + feat
        fused = self.fuse(fused)
        logits_p2 = self.pred(fused)
        logits = F.interpolate(
            logits_p2,
            size=(self.input_size, self.input_size),
            mode=self.upsample_mode,
            align_corners=False,
        )
        if validate:
            batch = fpn["p2"].shape[0]
            if logits.shape != (batch, 1, self.input_size, self.input_size):
                raise ValueError(f"segmentation logits shape {tuple(logits.shape)}")
        return {"logits": logits}
