"""96-channel lightweight FPN over C2–C5."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from auris.utils.shapes import expected_fpn_shapes


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        padding = kernel // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class LightweightFPN(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 96,
        input_size: int = 640,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("LightweightFPN expects 4 backbone levels (c2–c5).")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.input_size = input_size
        self.level_names = ["p2", "p3", "p4", "p5"]

        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )
        self.smooths = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, kernel=3) for _ in in_channels]
        )

    def forward(
        self, features: dict[str, torch.Tensor], validate: bool = True
    ) -> dict[str, torch.Tensor]:
        c2, c3, c4, c5 = features["c2"], features["c3"], features["c4"], features["c5"]
        l2, l3, l4, l5 = [lat(f) for lat, f in zip(self.laterals, (c2, c3, c4, c5))]

        p5 = l5
        p4 = l4 + F.interpolate(p5, size=l4.shape[-2:], mode="nearest")
        p3 = l3 + F.interpolate(p4, size=l3.shape[-2:], mode="nearest")
        p2 = l2 + F.interpolate(p3, size=l2.shape[-2:], mode="nearest")

        pyramids = [p2, p3, p4, p5]
        out = {
            name: smooth(feat)
            for name, smooth, feat in zip(self.level_names, self.smooths, pyramids)
        }
        if validate:
            batch = c2.shape[0]
            specs = expected_fpn_shapes(batch, self.input_size, self.out_channels)
            for name, spec in specs.items():
                spec.check(out[name])
        return out
