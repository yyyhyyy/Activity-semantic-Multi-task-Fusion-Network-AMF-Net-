# -*- coding: utf-8 -*-
"""Backbone 内部轻量适配器。

该文件提供可注入 VideoSwin/MViT 等 backbone 内部 block 的 residual adapter。
不是 head 后处理，也不是硬规则；adapter 被插入 backbone 子模块 forward 内部，参与端到端训练。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class InternalResidualAdapter(nn.Module):
    """通用内部 residual adapter，支持 token/feature map 张量。

    对最后一维通道做 bottleneck MLP：x + scale * Up(GELU(Down(LN(x))))。
    Swin3D/MViT 内部 block 常见输出为 (..., C)，因此直接适配最后一维。
    """

    def __init__(self, dim: int, reduction: int = 4, scale: float = 0.1, dropout: float = 0.0):
        super().__init__()
        hidden = max(8, int(dim) // max(int(reduction), 1))
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.norm.normalized_shape[0]:
            return x
        return x + self.scale * self.up(self.drop(self.act(self.down(self.norm(x)))))


class SpatioTemporalConvAdapter(nn.Module):
    """Lightweight channels-last spatio-temporal depthwise conv residual adapter.

    VideoSwin blocks often expose tensors as [B, T, H, W, C]. For these tensors,
    this adapter applies a depthwise 3D convolution over T/H/W and a pointwise
    projection. For token tensors that are not 5D, it falls back to the MLP
    residual adapter so injection remains safe.
    """

    def __init__(self, dim: int, reduction: int = 4, scale: float = 0.1, dropout: float = 0.0):
        super().__init__()
        hidden = max(8, int(dim) // max(int(reduction), 1))
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=dim, bias=False)
        self.pw = nn.Sequential(
            nn.Conv3d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, dim, kernel_size=1),
        )
        self.fallback = InternalResidualAdapter(dim=dim, reduction=reduction, scale=scale, dropout=dropout)
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.norm.normalized_shape[0]:
            return x
        if x.ndim != 5:
            return self.fallback(x)
        y = self.norm(x).permute(0, 4, 1, 2, 3).contiguous()
        y = self.pw(self.dw(y)).permute(0, 2, 3, 4, 1).contiguous()
        return x + self.scale * y


class EvidenceGatedSpatioTemporalAdapter(nn.Module):
    """Backbone-internal evidence-gated temporal adapter.

    It mixes short-range and dilated temporal depthwise convolutions inside
    VideoSwin/MViT blocks, then uses a channel gate to keep the residual
    selective. This is a backbone structure change, not a classifier-head rule.
    """

    def __init__(self, dim: int, reduction: int = 4, scale: float = 0.08, dropout: float = 0.0):
        super().__init__()
        hidden = max(8, int(dim) // max(int(reduction), 1))
        self.norm = nn.LayerNorm(dim)
        self.local_dw = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=dim, bias=False)
        self.temporal_dw = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1), groups=dim, bias=False)
        self.mix = nn.Sequential(
            nn.Conv3d(dim * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, dim, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.fallback = InternalResidualAdapter(dim=dim, reduction=reduction, scale=scale, dropout=dropout)
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.norm.normalized_shape[0]:
            return x
        if x.ndim != 5:
            return self.fallback(x)
        x_norm = self.norm(x)
        y = x_norm.permute(0, 4, 1, 2, 3).contiguous()
        mixed = self.mix(torch.cat([self.local_dw(y), self.temporal_dw(y)], dim=1))
        mixed = mixed.permute(0, 2, 3, 4, 1).contiguous()
        pooled = x_norm.mean(dim=(1, 2, 3))
        gate = self.gate(pooled).view(x.shape[0], 1, 1, 1, x.shape[-1])
        return x + torch.relu(self.scale) * gate * mixed


class AdapterWrappedBlock(nn.Module):
    """把 adapter 插入已有 backbone block 内部 forward 之后。"""

    def __init__(self, block: nn.Module, adapter: nn.Module):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return self.adapter(out)
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            return (self.adapter(out[0]), *out[1:])
        return out


def _infer_block_dim(block: nn.Module) -> int | None:
    """从常见 VideoSwin/MViT block 中推断通道维度。"""
    for attr in ("norm2", "norm1", "norm", "ln_2", "ln_1"):
        mod = getattr(block, attr, None)
        shape = getattr(mod, "normalized_shape", None)
        if shape:
            return int(shape[-1])
    for module in block.modules():
        if isinstance(module, nn.LayerNorm) and module.normalized_shape:
            return int(module.normalized_shape[-1])
    return None


def inject_internal_adapters(
    model: nn.Module,
    adapter: str = "none",
    reduction: int = 4,
    scale: float = 0.1,
    dropout: float = 0.0,
    max_blocks: int = 999,
) -> Tuple[nn.Module, int]:
    """递归注入内部 adapter。

    Parameters
    ----------
    model:
        待注入的 backbone。
    adapter:
        `none` 或 `ir_adapter`。
    max_blocks:
        最多注入多少个 block，用于控制参数量和速度。

    Returns
    -------
    model, injected_count
    """
    adapter = str(adapter).lower()
    if adapter in ("none", "", "identity"):
        return model, 0
    if adapter not in ("ir_adapter", "internal_residual", "mgba", "st_conv", "spatiotemporal_conv", "evidence_st", "evidence_st_conv"):
        raise ValueError(f"unsupported backbone adapter: {adapter}")

    injected = 0
    target_class_keywords = ("swintransformerblock", "shiftedwindowattention", "multiscaleattention", "multiscaleblock")

    def visit(parent: nn.Module):
        nonlocal injected
        for name, child in list(parent.named_children()):
            if injected >= max_blocks:
                return
            if isinstance(child, AdapterWrappedBlock):
                continue
            child_class = child.__class__.__name__.lower()
            has_nested = any(True for _ in child.named_children())
            is_real_block = any(k in child_class for k in target_class_keywords)
            dim = _infer_block_dim(child) if is_real_block else None
            if dim is not None and has_nested:
                if adapter in ("st_conv", "spatiotemporal_conv"):
                    adapter_module = SpatioTemporalConvAdapter(dim=dim, reduction=reduction, scale=scale, dropout=dropout)
                elif adapter in ("evidence_st", "evidence_st_conv"):
                    adapter_module = EvidenceGatedSpatioTemporalAdapter(dim=dim, reduction=reduction, scale=scale, dropout=dropout)
                else:
                    adapter_module = InternalResidualAdapter(dim=dim, reduction=reduction, scale=scale, dropout=dropout)
                setattr(
                    parent,
                    name,
                    AdapterWrappedBlock(
                        child,
                        adapter_module,
                    ),
                )
                injected += 1
            elif has_nested:
                visit(child)

    visit(model)
    return model, injected
