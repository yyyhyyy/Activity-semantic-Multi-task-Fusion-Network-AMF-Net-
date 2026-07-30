# -*- coding: utf-8 -*-
"""设备选择工具：支持 cpu / cuda / npu（Ascend）。"""

from __future__ import annotations

import torch


def get_device(device_arg: str) -> torch.device:
    """device_arg: 'cpu'|'cuda'|'npu'，或具体如 'cuda:0'/'npu:0'."""
    d = (device_arg or "").lower().strip()
    if d.startswith("npu"):
        # torch_npu 会注册 torch.npu
        try:
            import torch_npu  # noqa: F401
        except Exception as e:
            raise RuntimeError("你选择了 NPU，但当前环境未安装/未启用 torch_npu") from e
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu" if d == "npu" else d)
        return torch.device("cpu")

    if d.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda" if d == "cuda" else d)
        return torch.device("cpu")

    return torch.device("cpu")


def is_npu(device: torch.device) -> bool:
    return device.type == "npu"


def get_amp_components(device: torch.device, init_scale: float | None = None):
    """返回 (autocast_ctx, scaler_or_None)。

    - NPU: 优先使用 torch.npu.amp（需要 torch_npu）
    - CUDA: 使用 torch.cuda.amp
    - CPU: 不启用
    """
    if device.type == "npu":
        import torch_npu  # noqa: F401

        autocast = torch.npu.amp.autocast
        scaler_kwargs = {}
        if init_scale is not None and init_scale > 0:
            scaler_kwargs["init_scale"] = float(init_scale)
        scaler = torch.npu.amp.GradScaler(**scaler_kwargs)
        return autocast, scaler
    if device.type == "cuda":
        autocast = torch.cuda.amp.autocast
        scaler_kwargs = {}
        if init_scale is not None and init_scale > 0:
            scaler_kwargs["init_scale"] = float(init_scale)
        scaler = torch.cuda.amp.GradScaler(**scaler_kwargs)
        return autocast, scaler
    return None, None

