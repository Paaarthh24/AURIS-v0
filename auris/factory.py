"""Factories for model construction from YAML."""

from __future__ import annotations

from auris.models.auris import AURIS
from auris.utils.config import CfgNode, load_config


def build_model(cfg: CfgNode | str) -> AURIS:
    if not isinstance(cfg, CfgNode):
        cfg = load_config(cfg)
    return AURIS(cfg)
