"""ConvNeXt-inspired lightweight backbone.

Stages emit C2–C5 at strides 4/8/16/32 with channels [32, 64, 128, 256]
and depths [2, 2, 4, 2] by default. Each block is:

    residual + DropPath(
        LayerScale(
            PW↓ (GELU (PW↑ (LN (DWConv 7×7 (x))))))
        )
    )

Attention is applied after the residual as a registered hook (identity in v0).
"""

from __future__ import annotations

import torch
from torch import nn

from auris.models.hooks import build_attention
from auris.utils.shapes import ShapeError, expected_backbone_shapes


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors (ConvNeXt channels_first)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion_ratio: int = 4,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        attention_cfg=None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        hidden = dim * expansion_ratio
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=True
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.attention = build_attention(attention_cfg, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = residual + self.drop_path(x)
        return self.attention(x)


class ConvNeXtLiteBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | None = None,
        depths: list[int] | None = None,
        kernel_size: int = 7,
        expansion_ratio: int = 4,
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-6,
        stem_kernel: int = 4,
        stem_stride: int = 4,
        attention_cfg=None,
        input_size: int = 640,
    ) -> None:
        super().__init__()
        self.channels = list(channels or [32, 64, 128, 256])
        self.depths = list(depths or [2, 2, 4, 2])
        if len(self.channels) != 4 or len(self.depths) != 4:
            raise ValueError("AURIS v0 backbone requires 4 stages (channels and depths).")
        self.input_size = input_size
        self.out_names = ["c2", "c3", "c4", "c5"]

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_kernel,
                stride=stem_stride,
            ),
            LayerNorm2d(self.channels[0]),
        )

        total_blocks = sum(self.depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        block_id = 0
        for stage_i, (dim, depth) in enumerate(zip(self.channels, self.depths)):
            if stage_i == 0:
                self.downsamples.append(nn.Identity())
            else:
                prev = self.channels[stage_i - 1]
                self.downsamples.append(
                    nn.Sequential(
                        LayerNorm2d(prev),
                        nn.Conv2d(prev, dim, kernel_size=2, stride=2),
                    )
                )
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dim,
                        kernel_size=kernel_size,
                        expansion_ratio=expansion_ratio,
                        drop_path=dpr[block_id],
                        layer_scale_init=layer_scale_init,
                        attention_cfg=attention_cfg,
                    )
                )
                block_id += 1
            self.stages.append(nn.Sequential(*blocks))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, validate: bool = True) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ShapeError(f"backbone input must be NCHW, got {tuple(x.shape)}")
        features: dict[str, torch.Tensor] = {}
        x = self.stem(x)
        for name, down, stage in zip(self.out_names, self.downsamples, self.stages):
            x = down(x)
            x = stage(x)
            features[name] = x
        if validate:
            specs = expected_backbone_shapes(x.shape[0], self.input_size, self.channels)
            # batch comes from the last feature; re-read batch from stem output
            batch = features["c2"].shape[0]
            specs = expected_backbone_shapes(batch, self.input_size, self.channels)
            for name, spec in specs.items():
                spec.check(features[name])
        return features
