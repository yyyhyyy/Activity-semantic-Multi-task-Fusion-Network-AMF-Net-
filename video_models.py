# -*- coding: utf-8 -*-
"""视频多任务模型：3D ResNet backbone + 多任务分类头。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticDiscussHead(nn.Module):
    """融合多任务语义概率的 discuss_type 分类头。"""

    def __init__(self, feat_dim: int, semantic_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim + semantic_dim),
            nn.Linear(feat_dim + semantic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat: torch.Tensor, semantic_probs: list[torch.Tensor]) -> torch.Tensor:
        semantic = torch.cat(semantic_probs, dim=1)
        return self.net(torch.cat([feat, semantic], dim=1))


class VideoMultiTaskModel(nn.Module):
    """视频特征 + 多任务 head。

    输入：video (B, C, T, H, W)
    输出：dict(task_name -> logits (B, num_classes))
    """

    def __init__(
        self,
        num_classes_per_task: dict,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        dropout: float = 0.3,
        semantic_discuss_head: bool = False,
    ):
        super().__init__()
        self.task_names = list(num_classes_per_task.keys())
        self.num_classes_per_task = dict(num_classes_per_task)
        self.semantic_discuss_head = bool(semantic_discuss_head and "discuss_type" in self.num_classes_per_task)

        try:
            from torchvision.models.video import (
                r3d_18,
                R3D_18_Weights,
                mc3_18,
                MC3_18_Weights,
                r2plus1d_18,
                R2Plus1D_18_Weights,
            )
            # torchvision 新版提供的更强 backbone（若你的 torchvision 版本不含这些符号，会在下面 fallback）
            try:
                from torchvision.models.video import swin3d_t, Swin3D_T_Weights  # type: ignore
            except Exception:
                swin3d_t, Swin3D_T_Weights = None, None  # type: ignore
            try:
                from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights  # type: ignore
            except Exception:
                mvit_v2_s, MViT_V2_S_Weights = None, None  # type: ignore
        except Exception as e:
            raise RuntimeError("需要安装 torchvision 才能使用视频模型：pip install torchvision") from e

        if backbone == "r3d_18":
            weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
            base = r3d_18(weights=weights)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "mc3_18":
            weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
            base = mc3_18(weights=weights)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            base = r2plus1d_18(weights=weights)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "swin3d_t":
            if swin3d_t is None:
                raise RuntimeError("当前 torchvision 不支持 swin3d_t；请升级 torchvision。")
            weights = Swin3D_T_Weights.KINETICS400_V1 if (pretrained and Swin3D_T_Weights is not None) else None
            base = swin3d_t(weights=weights)
            # torchvision 的 Swin3D 通常是 base.head = Linear
            if not hasattr(base, "head"):
                raise RuntimeError("swin3d_t 模型结构不符合预期：缺少 head")
            feat_dim = base.head.in_features
            base.head = nn.Identity()
            self.backbone = base
        elif backbone == "mvit_v2_s":
            if mvit_v2_s is None:
                raise RuntimeError("当前 torchvision 不支持 mvit_v2_s；请升级 torchvision。")
            weights = MViT_V2_S_Weights.KINETICS400_V1 if (pretrained and MViT_V2_S_Weights is not None) else None
            base = mvit_v2_s(weights=weights)
            # torchvision 的 MViT v2 通常 base.head = Sequential(Dropout, Linear)
            if not hasattr(base, "head"):
                raise RuntimeError("mvit_v2_s 模型结构不符合预期：缺少 head")
            head = base.head
            if hasattr(head, "in_features"):
                feat_dim = head.in_features
            else:
                # 尝试从最后一层 Linear 推断
                last = None
                if isinstance(head, nn.Sequential) and len(head) > 0:
                    last = head[-1]
                if last is None or not hasattr(last, "in_features"):
                    raise RuntimeError("无法从 mvit_v2_s.head 推断 feat_dim")
                feat_dim = last.in_features
            base.head = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")

        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(feat_dim, n) for name, n in self.num_classes_per_task.items()})
        self.semantic_tasks = [
            name for name in ("scene_desk", "scene_method", "scene_inte", "teacher_act", "location", "stu_act", "view")
            if name in self.num_classes_per_task
        ]
        if self.semantic_discuss_head:
            semantic_dim = sum(int(self.num_classes_per_task[name]) for name in self.semantic_tasks)
            self.discuss_semantic_head = SemanticDiscussHead(
                feat_dim=feat_dim,
                semantic_dim=semantic_dim,
                num_classes=int(self.num_classes_per_task["discuss_type"]),
                hidden_dim=max(256, min(1024, feat_dim)),
                dropout=dropout,
            )
        else:
            self.discuss_semantic_head = None

    def forward(self, video: torch.Tensor) -> dict:
        feat = self.backbone(video)  # (B, feat_dim)
        feat = self.dropout(feat)
        logits = {name: self.heads[name](feat) for name in self.task_names}
        if self.discuss_semantic_head is not None and self.semantic_tasks:
            semantic_probs = [torch.softmax(logits[name].detach(), dim=1) for name in self.semantic_tasks]
            logits["discuss_type"] = self.discuss_semantic_head(feat, semantic_probs)
        return logits


def build_video_model(
    num_classes_per_task: dict,
    backbone: str = "r3d_18",
    pretrained: bool = True,
    semantic_discuss_head: bool = False,
) -> VideoMultiTaskModel:
    return VideoMultiTaskModel(
        num_classes_per_task=num_classes_per_task,
        backbone=backbone,
        pretrained=pretrained,
        dropout=0.3,
        semantic_discuss_head=semantic_discuss_head,
    )

