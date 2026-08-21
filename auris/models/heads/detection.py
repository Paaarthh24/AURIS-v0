"""Independent anchor-free detection head (FCOS-style ltrb + obj + cls)."""

from __future__ import annotations

import torch
from torch import nn

from auris.models.fpn import ConvBNAct
from auris.utils.shapes import spatial_from_input


class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 96,
        num_classes: int = 1,
        num_convs: int = 2,
        strides: list[int] | None = None,
        input_size: int = 640,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides or [4, 8, 16, 32])
        self.input_size = input_size

        stem = []
        for _ in range(num_convs):
            stem.append(ConvBNAct(in_channels, in_channels, kernel=3))
        self.stem = nn.Sequential(*stem) if stem else nn.Identity()
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.scales = nn.Parameter(torch.ones(len(self.strides)))

        for module in (self.cls_pred, self.reg_pred, self.obj_pred):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(
        self, fpn: dict[str, torch.Tensor], validate: bool = True
    ) -> dict[str, list[torch.Tensor]]:
        cls_maps, reg_maps, obj_maps = [], [], []
        for i, name in enumerate(["p2", "p3", "p4", "p5"]):
            feat = self.stem(fpn[name])
            cls_maps.append(self.cls_pred(feat))
            # softplus keeps ltrb distances positive and stable
            reg_maps.append(torch.nn.functional.softplus(self.scales[i] * self.reg_pred(feat)))
            obj_maps.append(self.obj_pred(feat))

        if validate:
            batch = fpn["p2"].shape[0]
            for i, stride in enumerate(self.strides):
                hw = spatial_from_input(self.input_size, stride)
                if cls_maps[i].shape != (batch, self.num_classes, hw, hw):
                    raise ValueError(f"cls level {i} shape {tuple(cls_maps[i].shape)}")
                if reg_maps[i].shape != (batch, 4, hw, hw):
                    raise ValueError(f"reg level {i} shape {tuple(reg_maps[i].shape)}")
                if obj_maps[i].shape != (batch, 1, hw, hw):
                    raise ValueError(f"obj level {i} shape {tuple(obj_maps[i].shape)}")

        return {"cls": cls_maps, "reg": reg_maps, "obj": obj_maps}
