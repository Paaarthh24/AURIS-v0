"""YAML configuration loading with nested attribute access."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class CfgNode:
    """Recursive dict wrapper: cfg.model.backbone.channels."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if data:
            for key, value in data.items():
                self._data[key] = _wrap(value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"Unknown config key: {name}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = _wrap(value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data[key] if key in self._data else default

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, CfgNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, CfgNode) else v for v in value]
            else:
                out[key] = value
        return out

    def clone(self) -> CfgNode:
        return CfgNode(deepcopy(self.to_dict()))

    def merge(self, other: Mapping[str, Any] | CfgNode) -> CfgNode:
        payload = other.to_dict() if isinstance(other, CfgNode) else dict(other)
        merged = _deep_merge(self.to_dict(), payload)
        return CfgNode(merged)

    def __repr__(self) -> str:
        return f"CfgNode({self.to_dict()})"


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return CfgNode(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> CfgNode:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    cfg = CfgNode(raw)
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg
