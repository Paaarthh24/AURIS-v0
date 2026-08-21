"""Parameter, MAC, FLOP, and latency profiling."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    total = 0
    trainable = 0
    named_roots = ["backbone", "fpn", "detection_head", "segmentation_head"]
    for root in named_roots:
        module = getattr(model, root, None)
        if module is None:
            breakdown[root] = 0
            continue
        n = sum(p.numel() for p in module.parameters())
        breakdown[root] = n
        total += n
        trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)
    leftover = sum(p.numel() for p in model.parameters()) - total
    breakdown["other"] = leftover
    breakdown["total"] = total + leftover
    breakdown["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return breakdown


def _conv_macs(mod: nn.Conv2d, inp: torch.Tensor, out: torch.Tensor) -> int:
    _, _, oh, ow = out.shape
    kh, kw = mod.kernel_size
    # MACs: k·k·Cin·Cout·H·W / groups
    return int(oh * ow * (mod.in_channels // mod.groups) * mod.out_channels * kh * kw)


def _linear_macs(mod: nn.Linear, inp: torch.Tensor, out: torch.Tensor) -> int:
    # out elements * in_features
    return int(out.numel() * mod.in_features)


def count_macs(model: nn.Module, dummy: torch.Tensor) -> dict[str, float]:
    macs_by_type: dict[str, int] = defaultdict(int)
    handles = []

    def conv_hook(mod, args, out):
        macs_by_type["conv"] += _conv_macs(mod, args[0], out)

    def linear_hook(mod, args, out):
        macs_by_type["linear"] += _linear_macs(mod, args[0], out)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(dummy, validate=False)
    for handle in handles:
        handle.remove()
    if was_training:
        model.train()

    conv = macs_by_type["conv"]
    linear = macs_by_type["linear"]
    total_macs = conv + linear
    return {
        "conv_macs": float(conv),
        "linear_macs": float(linear),
        "macs": float(total_macs),
        "flops": float(2 * total_macs),
        "gmacs": total_macs / 1e9,
        "gflops": (2 * total_macs) / 1e9,
    }


def profile_latency(
    model: nn.Module,
    dummy: torch.Tensor,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy, validate=False)
        if dummy.device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(dummy, validate=False)
            if dummy.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    return {
        "latency_ms": avg * 1000.0,
        "fps": 1.0 / avg if avg > 0 else 0.0,
        "warmup": float(warmup),
        "iters": float(iters),
        "device": str(dummy.device),
    }


def format_report(
    params: dict[str, int],
    macs: dict[str, float],
    latency: dict[str, float] | None,
    extra_lines: list[str] | None = None,
) -> str:
    lines = [
        "AURIS v0 computational report",
        "============================",
        f"Parameters (total):      {params['total']:,}",
        f"Parameters (trainable):  {params['trainable']:,}",
        f"  backbone:              {params['backbone']:,}",
        f"  fpn:                   {params['fpn']:,}",
        f"  detection_head:        {params['detection_head']:,}",
        f"  segmentation_head:     {params['segmentation_head']:,}",
        f"MACs:                    {macs['macs']:,.0f}  ({macs['gmacs']:.4f} G)",
        f"FLOPs (2*MAC):           {macs['flops']:,.0f}  ({macs['gflops']:.4f} G)",
        f"  conv MACs:             {macs['conv_macs']:,.0f}",
        f"  linear MACs:           {macs['linear_macs']:,.0f}",
    ]
    if latency:
        lines += [
            f"Latency:                 {latency['latency_ms']:.3f} ms  ({latency['device']})",
            f"Throughput:              {latency['fps']:.2f} FPS",
        ]
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def profile_model(model: nn.Module, input_size: int = 640, batch: int = 1, device: str = "cpu",
                  warmup: int = 10, iters: int = 50) -> dict[str, Any]:
    dummy = torch.randn(batch, 3, input_size, input_size, device=device)
    model = model.to(device)
    params = count_parameters(model)
    macs = count_macs(model, dummy)
    latency = profile_latency(model, dummy, warmup=warmup, iters=iters)
    report = format_report(params, macs, latency)
    return {"params": params, "macs": macs, "latency": latency, "report": report}
