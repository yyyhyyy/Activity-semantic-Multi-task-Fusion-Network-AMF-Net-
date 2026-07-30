# 文件说明：面向课堂 guide/debate 任务的 Swin3D-T 内部结构改造。
# -*- coding: utf-8 -*-
"""Classroom-adapted Swin3D-T backbone.

创新点（相对 torchvision swin3d_t）：
1. Stage Temporal Classroom Adapter (STCA)：在每个 Swin stage 输出后插入轻量时空卷积残差，强化帧间互动线索。
2. Pair-Discriminative Swin Head (PDSH)：用可学习的 guide/debate query 对时空 token 做注意力池化，
   输出 global_feat 与 pair_contrast = guide_token - debate_token 拼接特征，专门拉开 guide/debate。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassroomStageTemporalAdapter(nn.Module):
    """Factorized 3D conv residual adapter on stage feature maps."""

    def __init__(self, channels: int, scale: float = 0.1, dropout: float = 0.0):
        super().__init__()
        mid = max(8, channels // 8)
        self.channels = int(channels)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.temporal = nn.Conv3d(channels, mid, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.spatial = nn.Conv3d(mid, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            return x
        channel_last = x.shape[-1] == self.channels
        channel_first = x.shape[1] == self.channels
        if channel_last:
            x_cf = x.permute(0, 4, 1, 2, 3).contiguous()
        elif channel_first:
            x_cf = x
        else:
            return x
        y = self.drop(self.spatial(self.act(self.temporal(self.norm(x_cf)))))
        out = x_cf + self.scale * y
        if channel_last:
            return out.permute(0, 2, 3, 4, 1).contiguous()
        return out


class PairDiscriminativeSwinHead(nn.Module):
    """Dual-query attention pooling + global average for guide/debate contrast features."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = int(channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.guide_query = nn.Parameter(torch.randn(1, 1, self.channels) * 0.02)
        self.debate_query = nn.Parameter(torch.randn(1, 1, self.channels) * 0.02)
        self.attn = nn.MultiheadAttention(self.channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.channels)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.channels * 2),
            nn.Linear(self.channels * 2, self.channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _query_pool(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        q = query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 5 and x.shape[1] != self.channels and x.shape[-1] == self.channels:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        if x.dim() != 5:
            flat = x.flatten(1)
            return {
                "feat": flat,
                "global_feat": flat,
                "guide_token": flat,
                "debate_token": flat,
                "pair_contrast": flat.new_zeros((flat.shape[0], flat.shape[1])),
            }
        b, c, t, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        guide_tok = self._query_pool(tokens, self.guide_query)
        debate_tok = self._query_pool(tokens, self.debate_query)
        global_feat = self.pool(x).flatten(1)
        pair_contrast = guide_tok - debate_tok
        feat = self.proj(torch.cat([global_feat, pair_contrast], dim=1))
        return {
            "feat": feat,
            "global_feat": global_feat,
            "guide_token": guide_tok,
            "debate_token": debate_tok,
            "pair_contrast": pair_contrast,
        }


class ClassroomSwin3DBackbone(nn.Module):
    """Wrap torchvision swin3d_t with STCA + PDSH."""

    def __init__(self, swin_base: nn.Module, enable_stage_adapters: bool = True, adapter_scale: float = 0.1, adapter_dropout: float = 0.0):
        super().__init__()
        self.swin = swin_base
        stage_dim = self._infer_stage_dim()
        self.swin.head = nn.Identity()
        self.stage_adapters = nn.ModuleDict()
        self._register_stage_adapters(enable_stage_adapters, adapter_scale, adapter_dropout)
        self.pair_head = PairDiscriminativeSwinHead(stage_dim)
        self.out_dim = int(stage_dim)

    def _infer_stage_dim(self) -> int:
        head = getattr(self.swin, "head", None)
        if hasattr(head, "in_features"):
            return int(head.in_features)
        for module in self.swin.modules():
            if isinstance(module, nn.LayerNorm) and module.normalized_shape:
                return int(module.normalized_shape[0])
        return 768

    def _register_stage_adapters(self, enabled: bool, scale: float, dropout: float) -> None:
        if not enabled or not hasattr(self.swin, "features"):
            return
        stages = self.swin.features
        for idx, child in enumerate(stages):
            if not isinstance(child, nn.Sequential):
                continue
            dim = None
            for module in child.modules():
                if isinstance(module, nn.LayerNorm) and module.normalized_shape:
                    dim = int(module.normalized_shape[0])
            if dim is not None:
                self.stage_adapters[str(idx)] = ClassroomStageTemporalAdapter(dim, scale=scale, dropout=dropout)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.swin, "features"):
            return self.swin(x)
        out = x
        if hasattr(self.swin, "patch_embed") and out.dim() == 5 and out.shape[1] == 3:
            out = self.swin.patch_embed(out)
        if hasattr(self.swin, "pos_drop"):
            out = self.swin.pos_drop(out)
        for idx, layer in enumerate(self.swin.features):
            out = layer(out)
            key = str(idx)
            adapter = self.stage_adapters[key] if key in self.stage_adapters else None
            if adapter is not None:
                out = adapter(out)
        if hasattr(self.swin, "norm"):
            out = self.swin.norm(out)
        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self._forward_features(x)
        head_out = self.pair_head(feats)
        head_out["stage_feat"] = feats
        return head_out


def build_classroom_swin3d_t(pretrained: bool = True, enable_stage_adapters: bool = True, adapter_scale: float = 0.1, adapter_dropout: float = 0.0) -> Tuple[nn.Module, int]:
    from torchvision.models.video import Swin3D_T_Weights, swin3d_t

    base = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1 if pretrained else None)
    model = ClassroomSwin3DBackbone(
        base,
        enable_stage_adapters=enable_stage_adapters,
        adapter_scale=adapter_scale,
        adapter_dropout=adapter_dropout,
    )
    return model, int(model.out_dim)
