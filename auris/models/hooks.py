"""Pluggable attention, distillation, enhancement, and DA interfaces.

v0 ships identity / disabled implementations only. Enabling an unimplemented
type from YAML raises NotImplementedError so later modules can be dropped in
without changing the core graph.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class IdentityAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _not_implemented(name: str) -> Callable[..., nn.Module]:
    def factory(*_args, **_kwargs) -> nn.Module:
        raise NotImplementedError(
            f"Attention type '{name}' is reserved for a later AURIS version. "
            "Keep model.backbone.attention.enabled=false in v0."
        )

    return factory


ATTENTION_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "identity": lambda dim, **kwargs: IdentityAttention(),
    "se": _not_implemented("se"),
    "eca": _not_implemented("eca"),
    "coord": _not_implemented("coord"),
}


def build_attention(cfg, dim: int) -> nn.Module:
    enabled = bool(cfg.get("enabled", False)) if cfg is not None else False
    kind = str(cfg.get("type", "identity") or "identity") if cfg is not None else "identity"
    if not enabled:
        return IdentityAttention()
    if kind not in ATTENTION_REGISTRY:
        raise KeyError(f"Unknown attention type: {kind}")
    extra = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    extra.pop("enabled", None)
    extra.pop("type", None)
    return ATTENTION_REGISTRY[kind](dim, **extra)


def assert_future_modules_disabled(cfg) -> None:
    """Refuse to silently ignore unimplemented research modules."""
    model = cfg.model
    attn = model.backbone.attention
    if attn.enabled and attn.type not in (None, "identity"):
        build_attention(attn, dim=model.backbone.channels[0])

    if model.depth_estimation.enabled:
        raise NotImplementedError("Depth estimation is not part of AURIS v0.")
    if model.distillation.enabled:
        raise NotImplementedError("Knowledge distillation is not part of AURIS v0.")
    if model.domain_adaptation.enabled:
        raise NotImplementedError("Adversarial domain adaptation is not part of AURIS v0.")
    if cfg.data.underwater_enhancement.enabled:
        raise NotImplementedError("Underwater enhancement is not part of AURIS v0.")
