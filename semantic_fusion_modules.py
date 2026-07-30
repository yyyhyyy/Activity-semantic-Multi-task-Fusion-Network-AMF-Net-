# -*- coding: utf-8 -*-
"""可端到端学习的语义融合与类别不平衡损失模块。

本文件不包含硬规则。语义融合模块输入各任务 logits/概率，输出校正后的
`discuss_type` logits，可作为论文中的 BSF/LSF 模块进行消融。
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSemanticFusionMLP(nn.Module):
    """MLP 语义融合模块。

    输入：视频特征 + 所有辅助任务 logits/probabilities。
    输出：校正后的 discuss_type logits。
    """

    def __init__(
        self,
        feat_dim: int,
        semantic_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_feature: bool = True,
    ):
        super().__init__()
        self.use_feature = bool(use_feature)
        in_dim = semantic_dim + (feat_dim if self.use_feature else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, feat: torch.Tensor, semantic_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, semantic_vec], dim=1) if self.use_feature else semantic_vec
        return self.net(x)


class LearnableSemanticFusionAttention(nn.Module):
    """注意力式语义融合模块。

    将每个任务的 logits/prob 各自投影为 token，通过 MultiheadAttention 学习任务间关系，
    再与视频特征融合输出 discuss_type logits。
    """

    def __init__(
        self,
        feat_dim: int,
        task_dims: Iterable[int],
        num_classes: int,
        token_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_feature: bool = True,
    ):
        super().__init__()
        self.task_dims = [int(x) for x in task_dims]
        self.use_feature = bool(use_feature)
        self.proj = nn.ModuleList([nn.Linear(d, token_dim) for d in self.task_dims])
        self.attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        in_dim = token_dim + (feat_dim if self.use_feature else 0)
        self.cls = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, max(256, token_dim * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(256, token_dim * 2), num_classes),
        )

    def forward(self, feat: torch.Tensor, task_vectors: List[torch.Tensor]) -> torch.Tensor:
        tokens = [proj(vec) for proj, vec in zip(self.proj, task_vectors)]
        x = torch.stack(tokens, dim=1)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        pooled = self.norm(attn_out + x).mean(dim=1)
        fused = torch.cat([feat, pooled], dim=1) if self.use_feature else pooled
        return self.cls(fused)


def build_semantic_vectors(
    logits: Dict[str, torch.Tensor],
    task_names: List[str],
    mode: str = "prob",
    detach_aux: bool = False,
) -> List[torch.Tensor]:
    """从各任务 logits 构造融合输入。"""
    out = []
    for name in task_names:
        x = logits[name]
        if mode == "prob":
            x = F.softmax(x, dim=1)
        elif mode == "logit":
            x = x
        elif mode == "both":
            x = torch.cat([x, F.softmax(x, dim=1)], dim=1)
        else:
            raise ValueError(f"unknown semantic mode: {mode}")
        out.append(x.detach() if detach_aux else x)
    return out


class WeightedClassBalancedFocalLoss(nn.Module):
    """WCLS：类别均衡 focal loss，用于弱类增强消融。"""

    def __init__(self, class_weights: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()
