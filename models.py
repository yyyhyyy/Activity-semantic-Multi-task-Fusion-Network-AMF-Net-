# -*- coding: utf-8 -*-
"""多任务课堂行为分类模型：共享 backbone + 每任务一个分类头"""

import torch
import torch.nn as nn
from torchvision import models


class ClassroomMultiTaskModel(nn.Module):
    """
    共享 CNN backbone（ResNet18）+ 多个分类头。
    每个任务一个头，支持不同类别数。
    """

    def __init__(self, num_classes_per_task, backbone="resnet18", pretrained=True, dropout=0.3):
        super().__init__()
        self.task_names = list(num_classes_per_task.keys())
        self.num_classes_per_task = num_classes_per_task

        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        else:
            base = models.resnet18(weights=None)
            feat_dim = 512

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, n) for name, n in num_classes_per_task.items()
        })

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.dropout(feat)
        logits = {name: self.heads[name](feat) for name in self.task_names}
        return logits


def build_model(num_classes_per_task, backbone="resnet18", pretrained=True):
    return ClassroomMultiTaskModel(
        num_classes_per_task=num_classes_per_task,
        backbone=backbone,
        pretrained=pretrained,
        dropout=0.3,
    )
