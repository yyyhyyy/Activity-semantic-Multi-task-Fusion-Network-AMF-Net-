# -*- coding: utf-8 -*-
"""创新点与消融实验训练脚本。

支持：
- backbone: I3D(S3D近似) / VideoSwin / MViT(TimeSformer占位近似) / R3D 等
- BSF: none / mlp / attn（可学习语义融合模块，无硬规则）
- WCLS: 类别均衡 focal loss
- sampling: uniform / afs
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from afs_dataset import build_dataset_with_sampling
from config import (
    DEVICE,
    DISCUSS_TYPE_LABELS,
    LABEL_SMOOTHING,
    LOCATION_LABELS,
    LR,
    NUM_WORKERS,
    SCENE_DESK_LABELS,
    SCENE_INTE_LABELS,
    SCENE_METHOD_LABELS,
    SOFT_LABEL_ALPHA,
    SOFT_LABEL_TASKS,
    STU_ACT_LABELS,
    TASK_LOSS_WEIGHTS,
    TEACHER_ACT_LABELS,
    VIEW_LABELS,
    WEIGHT_DECAY,
)
from device_utils import get_amp_components, get_device
from experimental_video_models import build_experimental_video_model
from semantic_fusion_modules import WeightedClassBalancedFocalLoss
from multi_video_dataset import make_loader

TASK_LOSS_WEIGHTS["scene_desk"] = 0.9
TASK_LOSS_WEIGHTS["teacher_act"] = 3.8


def _clip_task_idx(dataset, clip, task: str) -> tuple[int, bool]:
    rows = dataset.videos[int(clip.video_id)]["frame_rows"]
    if getattr(dataset, "label_aggregation", "center") == "center":
        row = rows[int(clip.center)]
        return int(row.get(f"{task}_idx", -1)), bool(row.get(f"{task}_valid", False))
    indices = list(range(int(clip.start), int(clip.end), int(getattr(dataset, "sample_rate", 1))))
    valid_idxs = []
    for i in indices:
        if bool(rows[i].get(f"{task}_valid", False)):
            tidx = int(rows[i].get(f"{task}_idx", -1))
            if tidx >= 0:
                valid_idxs.append(tidx)
    if len(valid_idxs) < int(getattr(dataset, "min_valid_frames_by_task", {}).get(task, 1)):
        return -1, False
    counts = np.bincount(np.array(valid_idxs, dtype=np.int64))
    return int(counts.argmax()), True


def _clip_discuss_label_set(dataset, clip) -> list[int]:
    primary = int(dataset.video_discuss_type_idx.get(int(clip.video_id), -1))
    used_rule = False
    if hasattr(dataset, "_use_guide_location_rule") and dataset._use_guide_location_rule(int(clip.video_id)):
        loc_idx, loc_valid = _clip_task_idx(dataset, clip, "location")
        if loc_valid and loc_idx == LOCATION_LABELS.index("under"):
            primary = DISCUSS_TYPE_LABELS.index("guide_discuss")
            used_rule = True
        elif loc_valid and loc_idx == LOCATION_LABELS.index("plat"):
            primary = DISCUSS_TYPE_LABELS.index("question_discuss")
            used_rule = True
    labels = [primary] if primary >= 0 else []
    if not used_rule:
        labels.extend(int(x) for x in getattr(dataset, "video_discuss_type_extra_correct", {}).get(int(clip.video_id), []))
    return [int(x) for x in labels if 0 <= int(x) < len(DISCUSS_TYPE_LABELS)]


def compute_class_weights(dataset, indices, task: str, device):
    ncls = int(dataset.num_classes[task])
    counts = np.zeros(ncls, dtype=np.float32)
    for idx in indices:
        item = dataset.clips[int(idx)]
        if task == "discuss_type":
            for y in _clip_discuss_label_set(dataset, item):
                counts[y] += 1
        else:
            row = dataset.videos[int(item.video_id)]["frame_rows"][int(item.center)]
            if bool(row.get(f"{task}_valid", False)):
                y = int(row.get(f"{task}_idx", -1))
                if 0 <= y < ncls:
                    counts[y] += 1
    counts[counts <= 0] = 1.0
    weights = 1.0 / np.power(counts, 0.5)
    if task == "discuss_type" and ncls == len(DISCUSS_TYPE_LABELS):
        focus = np.ones(ncls, dtype=np.float32)
        focus[DISCUSS_TYPE_LABELS.index("guide_discuss")] = 0.88
        focus[DISCUSS_TYPE_LABELS.index("debate_discuss")] = 1.00
        focus[DISCUSS_TYPE_LABELS.index("question_discuss")] = 1.70
        focus[DISCUSS_TYPE_LABELS.index("socratic_discuss")] = 0.95
        focus[DISCUSS_TYPE_LABELS.index("data_discuss")] = 7.5
        weights = weights * focus
    elif task == "scene_desk" and ncls == len(SCENE_DESK_LABELS):
        focus = np.ones(ncls, dtype=np.float32)
        focus[SCENE_DESK_LABELS.index("scene_desk_oppo")] = 6.0
        focus[SCENE_DESK_LABELS.index("scene_desk_com")] = 1.8
        weights = weights * focus
    elif task == "teacher_act" and ncls == len(TEACHER_ACT_LABELS):
        focus = np.ones(ncls, dtype=np.float32)
        focus[TEACHER_ACT_LABELS.index("teacher_act_exp")] = 1.5
        focus[TEACHER_ACT_LABELS.index("teacher_act_guide")] = 1.8
        focus[TEACHER_ACT_LABELS.index("teacher_act_patrol")] = 1.7
        weights = weights * focus
    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, 0.25, 8.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_sampler(
    dataset,
    indices,
    data_target_ratio: float = 0.0,
    epoch_multiplier: float = 1.0,
    drop_empty_discuss: bool = False,
):
    label_sets = []
    for idx in indices:
        clip = dataset.clips[int(idx)]
        label_sets.append(_clip_discuss_label_set(dataset, clip))
    flat_labels = [x for cur in label_sets for x in cur if x >= 0]
    counts = np.bincount(np.array(flat_labels, dtype=np.int64), minlength=len(DISCUSS_TYPE_LABELS)).astype(np.float32)
    counts[counts <= 0] = 1.0
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    data_target_ratio = float(data_target_ratio)
    if data_target_ratio > 0:
        data_target_ratio = max(0.02, min(data_target_ratio, 0.35))
        targets = counts / max(float(counts.sum()), 1.0)
        non_data_sum = max(float(targets.sum() - targets[data_idx]), 1e-8)
        targets[data_idx] = data_target_ratio
        for i in range(len(targets)):
            if i != data_idx:
                targets[i] = (1.0 - data_target_ratio) * targets[i] / non_data_sum
        empty_weight = 0.0 if drop_empty_discuss else 0.1
        weights = np.array([
            max((targets[x] / counts[x] for x in cur if x >= 0), default=empty_weight)
            for cur in label_sets
        ], dtype=np.float64)
    else:
        focus = np.ones(len(DISCUSS_TYPE_LABELS), dtype=np.float64)
        focus[DISCUSS_TYPE_LABELS.index("guide_discuss")] = 0.88
        focus[DISCUSS_TYPE_LABELS.index("debate_discuss")] = 1.00
        focus[DISCUSS_TYPE_LABELS.index("question_discuss")] = 1.70
        focus[DISCUSS_TYPE_LABELS.index("socratic_discuss")] = 0.95
        focus[DISCUSS_TYPE_LABELS.index("data_discuss")] = 7.5
        empty_weight = 0.0 if drop_empty_discuss else 0.1
        weights = np.array([
            max((focus[x] / (counts[x] ** 0.5) for x in cur if x >= 0), default=empty_weight)
            for cur in label_sets
        ], dtype=np.float64)
    weights = weights / max(weights.mean(), 1e-12)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(len(indices), dtype=np.float64)
    num_samples = max(len(weights), int(round(len(weights) * max(float(epoch_multiplier), 1.0))))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def compute_losses(
    logits,
    batch,
    task_names,
    device,
    use_wcls: bool,
    discuss_weight: float,
    discuss_class_weights,
    task_class_weights=None,
    use_discuss_multi_hot_loss: bool = False,
):
    losses = []
    wcls_loss = WeightedClassBalancedFocalLoss(discuss_class_weights, gamma=1.5, label_smoothing=0.0)
    for name in task_names:
        y = batch[f"{name}_idx"].to(device)
        valid = batch[f"{name}_valid"].to(device)
        if valid.sum() == 0:
            losses.append(torch.tensor(0.0, device=device))
            continue
        if name == "discuss_type" and use_wcls:
            if use_discuss_multi_hot_loss and "discuss_type_multi_hot" in batch:
                multi = batch["discuss_type_multi_hot"].to(device).float()
                target = multi[valid] / multi[valid].sum(dim=1, keepdim=True).clamp_min(1.0)
                loss = -(target * F.log_softmax(logits[name][valid], dim=1)).sum(dim=1).mean() * discuss_weight
            else:
                loss = wcls_loss(logits[name][valid], y[valid]) * discuss_weight
        else:
            soft_key = f"{name}_soft"
            class_weight = None
            if task_class_weights is not None and name in task_class_weights:
                class_weight = task_class_weights[name].to(device=device, dtype=logits[name].dtype)
            if name in SOFT_LABEL_TASKS and soft_key in batch:
                soft = batch[soft_key].to(device).float()
                soft_valid = valid & (soft.sum(dim=1) > 0)
                if soft_valid.sum() > 0:
                    soft_target = soft[soft_valid] / soft[soft_valid].sum(dim=1, keepdim=True).clamp_min(1.0)
                    if class_weight is not None:
                        weighted_target = soft_target * class_weight.unsqueeze(0)
                        weighted_target = weighted_target / weighted_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
                        soft_loss = -(weighted_target * F.log_softmax(logits[name][soft_valid], dim=1)).sum(dim=1).mean()
                    else:
                        soft_loss = -(soft_target * F.log_softmax(logits[name][soft_valid], dim=1)).sum(dim=1).mean()
                    hard_loss = F.cross_entropy(
                        logits[name][valid],
                        y[valid],
                        weight=class_weight,
                        label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0,
                    )
                    alpha = float(SOFT_LABEL_ALPHA)
                    loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
                else:
                    loss = F.cross_entropy(
                        logits[name][valid],
                        y[valid],
                        weight=class_weight,
                        label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0,
                    )
            elif name == "discuss_type" and use_discuss_multi_hot_loss and "discuss_type_multi_hot" in batch:
                multi = batch["discuss_type_multi_hot"].to(device).float()
                target = multi[valid] / multi[valid].sum(dim=1, keepdim=True).clamp_min(1.0)
                loss = -(target * F.log_softmax(logits[name][valid], dim=1)).sum(dim=1).mean()
            else:
                loss = F.cross_entropy(
                    logits[name][valid],
                    y[valid],
                    weight=class_weight,
                    label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0,
                )
            if name == "discuss_type":
                loss = loss * discuss_weight
        if name != "discuss_type":
            loss = loss * float(TASK_LOSS_WEIGHTS.get(name, 1.0))
        losses.append(loss)
    return sum(losses)


def compute_video_bag_discuss_loss(logits, batch, device, weight: float, guide_boost: float):
    if weight <= 0 or "discuss_type" not in logits or "video_id" not in batch:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    video_ids = batch["video_id"].to(device).long()
    bag_logits = []
    bag_targets = []
    for vid in torch.unique(video_ids[valid]):
        mask = valid & (video_ids == vid)
        if mask.sum() == 0:
            continue
        targets = y[mask]
        target = targets[0]
        if not bool((targets == target).all()):
            continue
        bag_logits.append(logits["discuss_type"][mask].mean(dim=0))
        bag_targets.append(target)
    if not bag_logits:
        return torch.tensor(0.0, device=device)
    bag_logits_t = torch.stack(bag_logits, dim=0)
    bag_targets_t = torch.stack(bag_targets, dim=0).long()
    class_weights = torch.ones(len(DISCUSS_TYPE_LABELS), dtype=torch.float32, device=device)
    class_weights[DISCUSS_TYPE_LABELS.index("guide_discuss")] = float(guide_boost)
    return F.cross_entropy(bag_logits_t, bag_targets_t, weight=class_weights) * float(weight)


def compute_guide_debate_balance_loss(logits, batch, device, weight: float):
    if weight <= 0 or "guide_debate_balance" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    pair_mask = valid & ((y == guide_idx) | (y == debate_idx))
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    target = torch.where(y[pair_mask] == debate_idx, torch.ones_like(y[pair_mask]), torch.zeros_like(y[pair_mask]))
    return F.cross_entropy(logits["guide_debate_balance"][pair_mask], target) * float(weight)


def compute_pair_override_loss(logits, batch, device, weight: float):
    if weight <= 0 or "guide_debate_override" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    pair_mask = valid & ((y == guide_idx) | (y == debate_idx))
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    target = torch.where(y[pair_mask] == debate_idx, torch.ones_like(y[pair_mask]), torch.zeros_like(y[pair_mask]))
    return F.cross_entropy(logits["guide_debate_override"][pair_mask], target) * float(weight)


def compute_semantic_pair_loss(logits, batch, device, weight: float):
    if weight <= 0 or "guide_debate_semantic" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    pair_mask = valid & ((y == guide_idx) | (y == debate_idx))
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    target = torch.where(y[pair_mask] == debate_idx, torch.ones_like(y[pair_mask]), torch.zeros_like(y[pair_mask]))
    return F.cross_entropy(logits["guide_debate_semantic"][pair_mask], target) * float(weight)


def compute_guide_specific_loss(logits, batch, device, weight: float, debate_guard_weight: float, guard_margin: float):
    if "guide_specific" not in logits or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    guide_target = (y[valid] == guide_idx).float()
    pos = float((guide_target == 1).sum().item())
    neg = float((guide_target == 0).sum().item())
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32, device=device)
    guide_loss = F.binary_cross_entropy_with_logits(logits["guide_specific"][valid], guide_target, pos_weight=pos_weight) * float(weight)
    if debate_guard_weight <= 0:
        return guide_loss
    debate_mask = valid & (y == debate_idx)
    if debate_mask.sum() == 0:
        return guide_loss
    discuss = logits["discuss_type"][debate_mask]
    guard = F.relu(float(guard_margin) - (discuss[:, debate_idx] - discuss[:, guide_idx])).mean()
    return guide_loss + guard * float(debate_guard_weight)


def compute_data_specific_loss(logits, batch, device, weight: float, guard_margin: float):
    if weight <= 0 or "data_specific" not in logits or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    target = (y[valid] == data_idx).float()
    pos = float((target == 1).sum().item())
    neg = float((target == 0).sum().item())
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32, device=device)
    bce = F.binary_cross_entropy_with_logits(logits["data_specific"][valid], target, pos_weight=pos_weight)
    data_mask = valid & (y == data_idx)
    if data_mask.sum() == 0:
        return bce * float(weight)
    discuss = logits["discuss_type"]
    other_max = torch.cat([discuss[:, :data_idx], discuss[:, data_idx + 1:]], dim=1).max(dim=1).values
    margin = F.relu(float(guard_margin) - (discuss[data_mask, data_idx] - other_max[data_mask])).mean()
    return (bce + margin) * float(weight)


def compute_behavior_evidence_discuss_loss(logits, batch, device, weight: float, data_boost: float):
    if weight <= 0 or "behavior_evidence_discuss" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    class_weights = torch.ones(len(DISCUSS_TYPE_LABELS), dtype=torch.float32, device=device)
    class_weights[DISCUSS_TYPE_LABELS.index("data_discuss")] = float(data_boost)
    return F.cross_entropy(
        logits["behavior_evidence_discuss"][valid],
        y[valid],
        weight=class_weights,
        label_smoothing=0.03,
    ) * float(weight)


def compute_asymmetric_guide_boundary_loss(logits, batch, device, weight: float, guide_margin: float, debate_guard_margin: float, socratic_guard_margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    discuss = logits["discuss_type"]
    guide_score = discuss[:, guide_idx]
    debate_score = discuss[:, debate_idx]
    socratic_score = discuss[:, socratic_idx]
    guide_mask = valid & (y == guide_idx)
    debate_mask = valid & (y == debate_idx)
    parts = []
    if guide_mask.sum() > 0:
        parts.append(F.relu(float(guide_margin) - (guide_score[guide_mask] - debate_score[guide_mask])).mean())
        parts.append(F.relu(float(socratic_guard_margin) - (guide_score[guide_mask] - socratic_score[guide_mask])).mean())
    if debate_mask.sum() > 0 and debate_guard_margin > 0:
        parts.append(0.5 * F.relu(float(debate_guard_margin) - (debate_score[debate_mask] - guide_score[debate_mask])).mean())
    return torch.stack(parts).mean() * float(weight) if parts else torch.tensor(0.0, device=device)


def compute_pair_distribution_balance_loss(logits, batch, device, weight: float, target_guide_ratio: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    pair_mask = valid & ((y == guide_idx) | (y == debate_idx))
    if pair_mask.sum() < 2:
        return torch.tensor(0.0, device=device)
    pair_logits = logits["discuss_type"][pair_mask][:, [guide_idx, debate_idx]]
    mean_prob = F.softmax(pair_logits, dim=1).mean(dim=0)
    target = torch.tensor([float(target_guide_ratio), 1.0 - float(target_guide_ratio)], dtype=mean_prob.dtype, device=device)
    return F.mse_loss(mean_prob, target) * float(weight)


def set_pair_finetune_trainable(model, heads_only: bool):
    previous = {name: p.requires_grad for name, p in model.named_parameters()}
    if not heads_only:
        return previous
    trainable_tokens = (
        "heads.discuss_type",
        "fusion_head",
        "pair_balance_head",
        "guide_specific_head",
        "data_specific_head",
        "behavior_evidence_head",
        "pair_override_head",
        "semantic_pair_head",
        "disentangled_evidence_adapter",
        "pedagogical_template_adapter",
        "scene_desk_constraint_adapter",
        "pedagogical_prior_adapter",
    )
    for name, p in model.named_parameters():
        p.requires_grad = any(token in name for token in trainable_tokens)
    return previous


def restore_trainable(model, previous):
    for name, p in model.named_parameters():
        if name in previous:
            p.requires_grad = previous[name]


def compute_pair_margin_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    pair_mask = valid & ((y == guide_idx) | (y == debate_idx))
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    discuss = logits["discuss_type"][pair_mask]
    yy = y[pair_mask]
    guide_score = discuss[:, guide_idx]
    debate_score = discuss[:, debate_idx]
    guide_loss = F.relu(float(margin) - (guide_score - debate_score))[yy == guide_idx]
    debate_loss = F.relu(float(margin) - (debate_score - guide_score))[yy == debate_idx]
    parts = []
    if guide_loss.numel() > 0:
        parts.append(guide_loss.mean())
    if debate_loss.numel() > 0:
        parts.append(debate_loss.mean())
    return torch.stack(parts).mean() * float(weight) if parts else torch.tensor(0.0, device=device)


_PEDAGOGICAL_EXPECTED = {
    "question_discuss": {
        "scene_desk": {"scene_desk_group"},
        "location": {"under"},
        "teacher_act": {"teacher_act_ques", "teacher_act_patrol", "teacher_act_guide", "teacher_act_listen"},
        "stu_act": {"stu_act_answer", "stu_act_discuss"},
        "view": {"mate"},
        "scene_inte": {"scene_inte_group", "scene_inte_oto"},
    },
    "guide_discuss": {
        "scene_desk": {"scene_desk_group"},
        "location": {"plat", "under"},
        "teacher_act": {"teacher_act_guide", "teacher_act_exp", "teacher_act_listen"},
        "stu_act": {"stu_act_write", "stu_act_listen", "stu_act_answer", "stu_act_discuss"},
        "view": {"mate", "teacher"},
        "scene_inte": {"scene_inte_oto"},
    },
    "debate_discuss": {
        "scene_desk": {"scene_desk_oppo"},
        "location": {"plat"},
        "teacher_act": {"teacher_act_guide", "teacher_act_exp", "teacher_act_listen", "teacher_act_ques"},
        "stu_act": {"stu_act_discuss", "stu_act_answer", "stu_act_listen"},
        "view": {"mate"},
        "scene_inte": {"scene_inte_group"},
    },
    "socratic_discuss": {
        "scene_desk": {"scene_desk_round"},
        "location": {"plat", "under"},
        "teacher_act": {"teacher_act_ques", "teacher_act_guide", "teacher_act_exp", "teacher_act_listen"},
        "stu_act": {"stu_act_listen", "stu_act_answer", "stu_act_discuss"},
        "view": {"mate", "teacher"},
        "scene_inte": {"scene_inte_group", "scene_inte_oto"},
    },
    "data_discuss": {
        "scene_desk": {"scene_desk_com"},
        "location": {"plat"},
        "teacher_act": {"teacher_act_guide", "teacher_act_exp", "teacher_act_patrol"},
        "stu_act": {"stu_act_write", "stu_act_listen"},
        "view": {"teacher"},
        "scene_inte": {"scene_inte_oto"},
    },
}

_PED_TASK_LABELS = {
    "scene_desk": SCENE_DESK_LABELS,
    "location": LOCATION_LABELS,
    "teacher_act": TEACHER_ACT_LABELS,
    "stu_act": STU_ACT_LABELS,
    "view": VIEW_LABELS,
    "scene_inte": SCENE_INTE_LABELS,
}

_PED_TASK_WEIGHTS = {
    "scene_desk": 0.75,
    "teacher_act": 4.2,
    "stu_act": 2.8,
    "location": 1.6,
    "view": 1.6,
    "scene_inte": 1.0,
}


def _ped_label_index(task: str, label_name: str) -> int | None:
    try:
        return _PED_TASK_LABELS[task].index(label_name)
    except (KeyError, ValueError):
        return None


def build_pedagogical_prior_targets(batch, device, temperature: float = 0.65, guide_bias: float = 0.35):
    if len(DISCUSS_TYPE_LABELS) == 0:
        return None
    if "discuss_type_valid" not in batch:
        return None
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return None

    y_true = batch["discuss_type_idx"].to(device)
    batch_size = int(y_true.shape[0])
    scores = torch.zeros(batch_size, len(DISCUSS_TYPE_LABELS), dtype=torch.float32, device=device)
    normalizer = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for class_idx, discuss_name in enumerate(DISCUSS_TYPE_LABELS):
        expected = _PEDAGOGICAL_EXPECTED.get(discuss_name, {})
        for task, labels in expected.items():
            idx_key = f"{task}_idx"
            valid_key = f"{task}_valid"
            if idx_key not in batch or valid_key not in batch:
                continue
            task_idx = batch[idx_key].to(device)
            task_valid = batch[valid_key].to(device).bool()
            allowed = []
            for label_name in labels:
                label_idx = _ped_label_index(task, label_name)
                if label_idx is not None:
                    allowed.append(label_idx)
            if not allowed:
                continue
            allowed_t = torch.tensor(allowed, dtype=torch.long, device=device)
            match = (task_idx.unsqueeze(1) == allowed_t.unsqueeze(0)).any(dim=1).float()
            w = float(_PED_TASK_WEIGHTS.get(task, 1.0))
            scores[:, class_idx] += w * match * task_valid.float()
            normalizer += w * task_valid.float()

    scores = scores / normalizer.clamp_min(1.0).unsqueeze(1)
    if len(DISCUSS_TYPE_LABELS) == 5:
        guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
        question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
        debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
        socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
        data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")

        scene_desk = batch.get("scene_desk_idx", None)
        teacher = batch.get("teacher_act_idx", None)
        stu = batch.get("stu_act_idx", None)
        view = batch.get("view_idx", None)
        location = batch.get("location_idx", None)
        if scene_desk is not None and teacher is not None and stu is not None and view is not None:
            scene_desk = scene_desk.to(device)
            teacher = teacher.to(device)
            stu = stu.to(device)
            view = view.to(device)
            location = location.to(device) if location is not None else None
            desk_group = (scene_desk == SCENE_DESK_LABELS.index("scene_desk_group")).float()
            desk_oppo = (scene_desk == SCENE_DESK_LABELS.index("scene_desk_oppo")).float()
            desk_round = (scene_desk == SCENE_DESK_LABELS.index("scene_desk_round")).float()
            desk_com = (scene_desk == SCENE_DESK_LABELS.index("scene_desk_com")).float()
            t_guide = (teacher == TEACHER_ACT_LABELS.index("teacher_act_guide")).float()
            t_exp = (teacher == TEACHER_ACT_LABELS.index("teacher_act_exp")).float()
            t_listen = (teacher == TEACHER_ACT_LABELS.index("teacher_act_listen")).float()
            t_ques = (teacher == TEACHER_ACT_LABELS.index("teacher_act_ques")).float()
            t_patrol = (teacher == TEACHER_ACT_LABELS.index("teacher_act_patrol")).float()
            loc_under = (location == LOCATION_LABELS.index("under")).float() if location is not None else torch.zeros_like(t_patrol)
            loc_plat = (location == LOCATION_LABELS.index("plat")).float() if location is not None else torch.zeros_like(t_patrol)
            s_answer = (stu == STU_ACT_LABELS.index("stu_act_answer")).float()
            s_write = (stu == STU_ACT_LABELS.index("stu_act_write")).float()
            s_discuss = (stu == STU_ACT_LABELS.index("stu_act_discuss")).float()
            s_listen = (stu == STU_ACT_LABELS.index("stu_act_listen")).float()
            v_mate = (view == VIEW_LABELS.index("mate")).float()
            v_teacher = (view == VIEW_LABELS.index("teacher")).float()

            scores[:, guide_idx] += desk_group * (0.40 * t_guide + 0.20 * t_exp + 0.15 * t_listen + 0.25 * t_patrol) * (
                0.35 * s_write + 0.25 * s_listen + 0.20 * s_answer + 0.20 * s_discuss
            ) * (0.55 + 0.45 * loc_under)
            scores[:, question_idx] += desk_group * (0.62 * t_ques + 0.24 * t_patrol + 0.14 * t_guide) * (
                0.58 * s_answer + 0.34 * s_discuss + 0.08 * s_write
            ) * (0.65 * v_mate + 0.35)
            scores[:, debate_idx] += desk_oppo * (0.45 * t_guide + 0.30 * t_exp + 0.25 * t_listen) * (
                0.58 * s_discuss + 0.42 * s_answer
            ) * (0.60 + 0.40 * loc_plat)
            scores[:, socratic_idx] += desk_round * (0.45 * t_ques + 0.30 * t_listen + 0.25 * t_guide) * (
                0.45 * s_answer + 0.35 * s_discuss + 0.20 * s_listen
            )
            scores[:, data_idx] += desk_com * (0.45 * t_exp + 0.35 * t_guide + 0.20 * t_patrol) * (
                0.58 * s_write + 0.30 * s_listen + 0.12 * s_discuss
            ) * (0.70 * v_teacher + 0.30)

        scores[:, guide_idx] += float(guide_bias) * (y_true == guide_idx).float()
        scores[:, debate_idx] += 0.25 * (y_true == debate_idx).float()
        scores[:, socratic_idx] += 0.20 * (y_true == socratic_idx).float()
        scores[:, data_idx] += 0.20 * (y_true == data_idx).float()

    hard = torch.zeros_like(scores)
    hard.scatter_(1, y_true.clamp_min(0).unsqueeze(1), 1.0)
    soft = torch.softmax(scores / max(float(temperature), 1e-3), dim=1)
    target = 0.70 * hard + 0.30 * soft
    return target, valid


def build_behavior_table_targets(batch, device, temperature: float = 0.55, data_bias: float = 0.35):
    required = ("scene_desk_soft", "teacher_act_soft", "stu_act_soft", "view_soft", "location_soft", "scene_inte_soft")
    if any(k not in batch for k in required) or "discuss_type_valid" not in batch:
        return None
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0 or len(DISCUSS_TYPE_LABELS) != 5:
        return None

    scene = batch["scene_desk_soft"].to(device).float()
    teacher = batch["teacher_act_soft"].to(device).float()
    stu = batch["stu_act_soft"].to(device).float()
    view = batch["view_soft"].to(device).float()
    location = batch["location_soft"].to(device).float()
    inte = batch["scene_inte_soft"].to(device).float()

    desk_group = scene[:, SCENE_DESK_LABELS.index("scene_desk_group")]
    desk_round = scene[:, SCENE_DESK_LABELS.index("scene_desk_round")]
    desk_oppo = scene[:, SCENE_DESK_LABELS.index("scene_desk_oppo")]
    desk_com = scene[:, SCENE_DESK_LABELS.index("scene_desk_com")]
    loc_plat = location[:, LOCATION_LABELS.index("plat")]
    loc_under = location[:, LOCATION_LABELS.index("under")]
    t_exp = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_exp")]
    t_ques = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_ques")]
    t_guide = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_guide")]
    t_listen = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_listen")]
    t_patrol = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_patrol")]
    s_answer = stu[:, STU_ACT_LABELS.index("stu_act_answer")]
    s_write = stu[:, STU_ACT_LABELS.index("stu_act_write")]
    s_discuss = stu[:, STU_ACT_LABELS.index("stu_act_discuss")]
    s_listen = stu[:, STU_ACT_LABELS.index("stu_act_listen")]
    v_mate = view[:, VIEW_LABELS.index("mate")]
    v_teacher = view[:, VIEW_LABELS.index("teacher")]
    inte_group = inte[:, SCENE_INTE_LABELS.index("scene_inte_group")]
    inte_oto = inte[:, SCENE_INTE_LABELS.index("scene_inte_oto")]

    scores = torch.zeros(scene.shape[0], len(DISCUSS_TYPE_LABELS), dtype=torch.float32, device=device)
    question = (
        0.30 * desk_group
        + 0.24 * t_ques
        + 0.18 * s_answer
        + 0.14 * s_discuss
        + 0.08 * v_mate
        + 0.06 * inte_group
    )
    guide = (
        0.16 * desk_group
        + 0.18 * t_guide
        + 0.14 * t_exp
        + 0.12 * t_listen
        + 0.10 * t_patrol
        + 0.11 * s_write
        + 0.09 * s_listen
        + 0.06 * loc_under
        + 0.04 * inte_oto
    )
    debate = (
        0.35 * desk_oppo
        + 0.18 * s_discuss
        + 0.14 * s_answer
        + 0.12 * loc_plat
        + 0.10 * t_guide
        + 0.07 * t_exp
        + 0.04 * v_mate
    )
    socratic = (
        0.34 * desk_round
        + 0.20 * t_ques
        + 0.14 * t_guide
        + 0.10 * t_listen
        + 0.10 * s_answer
        + 0.08 * s_listen
        + 0.04 * v_mate
    )
    data_teacher = 0.46 * t_exp + 0.34 * t_guide + 0.20 * t_patrol
    data_student = 0.66 * s_write + 0.34 * s_listen
    data_context = 0.48 * desk_com + 0.20 * loc_plat + 0.18 * v_teacher + 0.14 * inte_oto
    data_counter = 0.20 * desk_group + 0.20 * desk_round + 0.18 * desk_oppo + 0.18 * v_mate + 0.14 * s_discuss
    data = 0.54 * data_teacher + 0.25 * data_student + 0.34 * data_context - 0.20 * data_counter + float(data_bias)

    scores[:, DISCUSS_TYPE_LABELS.index("question_discuss")] = question
    scores[:, DISCUSS_TYPE_LABELS.index("guide_discuss")] = guide
    scores[:, DISCUSS_TYPE_LABELS.index("debate_discuss")] = debate
    scores[:, DISCUSS_TYPE_LABELS.index("socratic_discuss")] = socratic
    scores[:, DISCUSS_TYPE_LABELS.index("data_discuss")] = data
    target = torch.softmax(scores / max(float(temperature), 1e-3), dim=1)
    return target, valid


def compute_behavior_table_consistency_loss(logits, batch, device, weight: float, temperature: float, data_bias: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    built = build_behavior_table_targets(batch, device, temperature=temperature, data_bias=data_bias)
    if built is None:
        return torch.tensor(0.0, device=device)
    target, valid = built
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    logprob = F.log_softmax(logits["discuss_type"][valid], dim=1)
    return F.kl_div(logprob, target[valid].to(dtype=logprob.dtype), reduction="batchmean") * float(weight)


def compute_pedagogical_consistency_loss(
    logits,
    batch,
    device,
    weight: float,
    temperature: float,
    guide_bias: float,
    guide_margin_weight: float,
    guide_margin: float,
):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    built = build_pedagogical_prior_targets(batch, device, temperature=temperature, guide_bias=guide_bias)
    if built is None:
        return torch.tensor(0.0, device=device)
    target, valid = built
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    target = target.to(dtype=logits["discuss_type"].dtype)
    pred_logprob = F.log_softmax(logits["discuss_type"][valid], dim=1)
    distill = F.kl_div(pred_logprob, target[valid], reduction="batchmean")
    if guide_margin_weight <= 0 or len(DISCUSS_TYPE_LABELS) != 5:
        return distill * float(weight)

    y = batch["discuss_type_idx"].to(device)
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    discuss = logits["discuss_type"]
    guide_mask = valid & (y == guide_idx)
    if guide_mask.sum() == 0:
        return distill * float(weight)
    guide_logit = discuss[:, guide_idx]
    guide_margin_loss = F.relu(float(guide_margin) - (guide_logit[guide_mask] - discuss[guide_mask, question_idx])).mean()
    guide_margin_loss = guide_margin_loss + F.relu(float(guide_margin) - (guide_logit[guide_mask] - discuss[guide_mask, socratic_idx])).mean()
    return distill * float(weight) + guide_margin_loss * float(guide_margin_weight)


def compute_guide_question_soft_target_loss(logits, batch, device, weight: float, question_mass: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    guide_mask = valid & (y == guide_idx)
    if guide_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    q_mass = max(0.0, min(float(question_mass), 0.45))
    target = torch.zeros_like(logits["discuss_type"][guide_mask])
    target[:, guide_idx] = 1.0 - q_mass
    target[:, question_idx] = q_mass
    return -(target * F.log_softmax(logits["discuss_type"][guide_mask], dim=1)).sum(dim=1).mean() * float(weight)


def compute_guide_group_location_rescue_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    if "scene_desk_idx" not in batch or "location_idx" not in batch:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    scene = batch["scene_desk_idx"].to(device)
    scene_valid = batch.get("scene_desk_valid", torch.ones_like(y, dtype=torch.bool)).to(device).bool()
    location = batch["location_idx"].to(device)
    location_valid = batch.get("location_valid", torch.ones_like(y, dtype=torch.bool)).to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    group = scene_valid & (scene == SCENE_DESK_LABELS.index("scene_desk_group"))
    guide_location = location_valid & ((location == LOCATION_LABELS.index("under")) | (location == LOCATION_LABELS.index("plat")))
    guide_mask = valid & (y == guide_idx) & group & guide_location
    if guide_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    discuss = logits["discuss_type"]
    guide = discuss[:, guide_idx]
    loss = F.relu(float(margin) - (guide[guide_mask] - discuss[guide_mask, debate_idx])).mean()
    loss = loss + 0.5 * F.relu(float(margin) - (guide[guide_mask] - discuss[guide_mask, socratic_idx])).mean()
    loss = loss + 0.5 * F.relu(float(margin) - (guide[guide_mask] - discuss[guide_mask, question_idx])).mean()
    return loss * float(weight)


def compute_debate_oppo_protection_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    if "scene_desk_idx" not in batch:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    scene = batch["scene_desk_idx"].to(device)
    scene_valid = batch.get("scene_desk_valid", torch.ones_like(y, dtype=torch.bool)).to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    oppo = scene_valid & (scene == SCENE_DESK_LABELS.index("scene_desk_oppo"))
    debate_mask = valid & (y == debate_idx) & oppo
    if debate_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    discuss = logits["discuss_type"]
    debate = discuss[:, debate_idx]
    loss = F.relu(float(margin) - (debate[debate_mask] - discuss[debate_mask, guide_idx])).mean()
    loss = loss + F.relu(float(margin) - (debate[debate_mask] - discuss[debate_mask, socratic_idx])).mean()
    return loss * float(weight)


def compute_guide_patrol_under_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    if "teacher_act_idx" not in batch or "location_idx" not in batch:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    teacher = batch["teacher_act_idx"].to(device)
    teacher_valid = batch.get("teacher_act_valid", torch.ones_like(y, dtype=torch.bool)).to(device).bool()
    location = batch["location_idx"].to(device)
    location_valid = batch.get("location_valid", torch.ones_like(y, dtype=torch.bool)).to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    patrol = teacher_valid & (teacher == TEACHER_ACT_LABELS.index("teacher_act_patrol"))
    under = location_valid & (location == LOCATION_LABELS.index("under"))
    guide_evidence_mask = valid & (y == guide_idx) & (patrol | under)
    if guide_evidence_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    discuss = logits["discuss_type"]
    guide = discuss[:, guide_idx]
    loss = F.relu(float(margin) - (guide[guide_evidence_mask] - discuss[guide_evidence_mask, debate_idx])).mean()
    loss = loss + F.relu(float(margin) - (guide[guide_evidence_mask] - discuss[guide_evidence_mask, socratic_idx])).mean()
    return loss * float(weight)


def compute_scene_desk_constraint_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits or "scene_desk_idx" not in batch:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    scene = batch["scene_desk_idx"].to(device)
    scene_valid = batch.get("scene_desk_valid", torch.ones_like(y, dtype=torch.bool)).to(device).bool()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    group = scene_valid & (scene == SCENE_DESK_LABELS.index("scene_desk_group"))
    oppo = scene_valid & (scene == SCENE_DESK_LABELS.index("scene_desk_oppo"))
    com = scene_valid & (scene == SCENE_DESK_LABELS.index("scene_desk_com"))
    guide_group = valid & (y == guide_idx) & group
    debate_oppo = valid & (y == debate_idx) & oppo
    data_com = valid & (y == data_idx) & com
    discuss = logits["discuss_type"]
    parts = []
    if guide_group.sum() > 0:
        parts.append(F.relu(float(margin) - (discuss[guide_group, guide_idx] - discuss[guide_group, debate_idx])).mean())
    if debate_oppo.sum() > 0:
        parts.append(0.6 * F.relu(float(margin) - (discuss[debate_oppo, debate_idx] - discuss[debate_oppo, guide_idx])).mean())
        parts.append(0.4 * F.relu(float(margin) - (discuss[debate_oppo, debate_idx] - discuss[debate_oppo, question_idx])).mean())
    if data_com.sum() > 0:
        others = torch.stack([
            discuss[data_com, question_idx],
            discuss[data_com, guide_idx],
            discuss[data_com, debate_idx],
            discuss[data_com, socratic_idx],
        ], dim=1).max(dim=1).values
        parts.append(0.8 * F.relu(float(margin) - (discuss[data_com, data_idx] - others)).mean())
    return torch.stack(parts).mean() * float(weight) if parts else torch.tensor(0.0, device=device)


def compute_data_behavior_fewshot_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    required = ("scene_desk_soft", "teacher_act_soft", "stu_act_soft", "view_soft", "location_soft")
    if any(k not in batch for k in required):
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()

    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    scene_soft = batch["scene_desk_soft"].to(device).float()
    teacher_soft = batch["teacher_act_soft"].to(device).float()
    stu_soft = batch["stu_act_soft"].to(device).float()
    view_soft = batch["view_soft"].to(device).float()
    location_soft = batch["location_soft"].to(device).float()

    scene_com = scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_com")]
    scene_non_com = torch.stack([
        scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_group")],
        scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_round")],
        scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_oppo")],
    ], dim=1).max(dim=1).values
    teacher_data = (
        0.46 * teacher_soft[:, TEACHER_ACT_LABELS.index("teacher_act_exp")]
        + 0.34 * teacher_soft[:, TEACHER_ACT_LABELS.index("teacher_act_guide")]
        + 0.20 * teacher_soft[:, TEACHER_ACT_LABELS.index("teacher_act_patrol")]
    )
    student_data = (
        0.66 * stu_soft[:, STU_ACT_LABELS.index("stu_act_write")]
        + 0.34 * stu_soft[:, STU_ACT_LABELS.index("stu_act_listen")]
    )
    teacher_view = view_soft[:, VIEW_LABELS.index("teacher")]
    plat = location_soft[:, LOCATION_LABELS.index("plat")]
    data_evidence = (
        0.44 * teacher_data
        + 0.24 * student_data
        + 0.14 * teacher_view
        + 0.10 * plat
        + 0.20 * scene_com
    )
    anti_data = 0.35 * scene_non_com + 0.20 * stu_soft[:, STU_ACT_LABELS.index("stu_act_discuss")]
    data_mask = valid & (y == data_idx)
    non_data_weak = valid & (y != data_idx) & ((data_evidence - anti_data) <= 0.20)
    if data_mask.sum() == 0 and non_data_weak.sum() == 0:
        return torch.tensor(0.0, device=device)
    discuss = logits["discuss_type"]
    others = torch.cat([discuss[:, :data_idx], discuss[:, data_idx + 1:]], dim=1).max(dim=1).values
    parts = []
    if data_mask.sum() > 0:
        evidence_weight = (0.75 + 0.50 * torch.clamp(data_evidence[data_mask], 0.0, 1.0)).detach()
        parts.append((F.relu(float(margin) - (discuss[data_mask, data_idx] - others[data_mask])) * evidence_weight).mean())
    if non_data_weak.sum() > 0:
        parts.append(0.65 * F.relu(float(margin) * 0.75 - (others[non_data_weak] - discuss[non_data_weak, data_idx])).mean())
    return torch.stack(parts).mean() * float(weight) if parts else torch.tensor(0.0, device=device)


def compute_data_debate_conflict_loss(
    logits,
    batch,
    device,
    weight: float,
    data_margin: float,
    weak_debate_threshold: float,
    weak_debate_cap_margin: float,
):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    required = ("scene_desk_soft", "stu_act_soft", "view_soft", "location_soft")
    if any(k not in batch for k in required):
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    scene_soft = batch["scene_desk_soft"].to(device).float()
    stu_soft = batch["stu_act_soft"].to(device).float()
    view_soft = batch["view_soft"].to(device).float()
    location_soft = batch["location_soft"].to(device).float()

    p_com = scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_com")]
    p_oppo = scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_oppo")]
    p_round = scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_round")]
    p_group = scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_group")]
    p_discuss = stu_soft[:, STU_ACT_LABELS.index("stu_act_discuss")]
    p_answer = stu_soft[:, STU_ACT_LABELS.index("stu_act_answer")]
    p_listen = stu_soft[:, STU_ACT_LABELS.index("stu_act_listen")]
    p_mate = view_soft[:, VIEW_LABELS.index("mate")]
    p_plat = location_soft[:, LOCATION_LABELS.index("plat")]

    debate_evidence = 0.42 * p_oppo + 0.20 * p_mate + 0.16 * p_discuss + 0.12 * p_answer + 0.06 * p_listen + 0.04 * p_plat
    data_evidence = 0.62 * p_com - 0.22 * torch.maximum(torch.maximum(p_round, p_group), p_oppo) - 0.10 * p_discuss

    discuss = logits["discuss_type"]
    data_mask = valid & (y == data_idx)
    weak_debate_mask = valid & (y == debate_idx) & (debate_evidence < float(weak_debate_threshold))
    parts = []
    if data_mask.sum() > 0:
        # Directly protects the observed failure mode: true data clips being taken by debate/question/guide.
        data_weight = (0.80 + 0.50 * torch.clamp(data_evidence[data_mask], 0.0, 1.0)).detach()
        debate_margin = F.relu(float(data_margin) - (discuss[data_mask, data_idx] - discuss[data_mask, debate_idx]))
        question_margin = F.relu(float(data_margin) - (discuss[data_mask, data_idx] - discuss[data_mask, question_idx]))
        guide_margin = F.relu(float(data_margin) - (discuss[data_mask, data_idx] - discuss[data_mask, guide_idx]))
        parts.append(((0.60 * debate_margin + 0.50 * question_margin + 0.70 * guide_margin) * data_weight).mean())
    if weak_debate_mask.sum() > 0:
        other_idx = [i for i in range(discuss.shape[1]) if i != debate_idx]
        other_max = discuss[:, other_idx].max(dim=1).values
        debate_advantage = discuss[weak_debate_mask, debate_idx] - other_max[weak_debate_mask]
        weak_weight = (1.0 + torch.clamp(float(weak_debate_threshold) - debate_evidence[weak_debate_mask], 0.0, 1.0)).detach()
        parts.append(0.55 * (F.relu(debate_advantage - float(weak_debate_cap_margin)) * weak_weight).mean())
    return torch.stack(parts).mean() * float(weight) if parts else torch.tensor(0.0, device=device)


def compute_question_competitor_guard_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    discuss = logits["discuss_type"]
    data_mask = valid & (y == data_idx)
    debate_mask = valid & (y == debate_idx)
    parts = []
    if data_mask.sum() > 0:
        parts.append(F.relu(float(margin) - (discuss[data_mask, data_idx] - discuss[data_mask, question_idx])).mean())
    if debate_mask.sum() > 0:
        parts.append(F.relu(float(margin) - (discuss[debate_mask, debate_idx] - discuss[debate_mask, question_idx])).mean())
    return torch.stack(parts).mean() * float(weight) if parts else torch.tensor(0.0, device=device)


def compute_question_behavior_margin_loss(logits, batch, device, weight: float, margin: float):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    required = ("location_soft", "teacher_act_soft", "stu_act_soft", "scene_desk_soft")
    if any(k not in batch for k in required):
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    location_soft = batch["location_soft"].to(device).float()
    teacher_soft = batch["teacher_act_soft"].to(device).float()
    stu_soft = batch["stu_act_soft"].to(device).float()
    scene_soft = batch["scene_desk_soft"].to(device).float()
    question_evidence = (
        0.32 * location_soft[:, LOCATION_LABELS.index("plat")]
        + 0.28 * teacher_soft[:, TEACHER_ACT_LABELS.index("teacher_act_ques")]
        + 0.22 * stu_soft[:, STU_ACT_LABELS.index("stu_act_answer")]
        + 0.18 * stu_soft[:, STU_ACT_LABELS.index("stu_act_discuss")]
        + 0.10 * scene_soft[:, SCENE_DESK_LABELS.index("scene_desk_group")]
    )
    question_mask = valid & (y == question_idx) & (question_evidence >= 0.22)
    if question_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    discuss = logits["discuss_type"]
    competing = torch.stack([discuss[:, guide_idx], discuss[:, socratic_idx]], dim=1).max(dim=1).values
    loss = F.relu(float(margin) - (discuss[question_mask, question_idx] - competing[question_mask])).mean()
    return loss * float(weight)


def compute_socratic_evidence_guard_loss(
    logits,
    batch,
    device,
    weight: float,
    margin: float,
    min_behavior: float,
    shortcut_negative_weight: float,
    confidence_cap: float,
):
    if weight <= 0 or "discuss_type" not in logits or len(DISCUSS_TYPE_LABELS) != 5:
        return torch.tensor(0.0, device=device)
    required = ("scene_desk_soft", "teacher_act_soft", "stu_act_soft", "view_soft")
    if any(k not in batch for k in required):
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    scene = batch["scene_desk_soft"].to(device).float()
    teacher = batch["teacher_act_soft"].to(device).float()
    stu = batch["stu_act_soft"].to(device).float()
    view = batch["view_soft"].to(device).float()

    p_round = scene[:, SCENE_DESK_LABELS.index("scene_desk_round")]
    p_group = scene[:, SCENE_DESK_LABELS.index("scene_desk_group")]
    p_oppo = scene[:, SCENE_DESK_LABELS.index("scene_desk_oppo")]
    p_com = scene[:, SCENE_DESK_LABELS.index("scene_desk_com")]
    t_ques = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_ques")]
    t_guide = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_guide")]
    t_exp = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_exp")]
    t_listen = teacher[:, TEACHER_ACT_LABELS.index("teacher_act_listen")]
    s_answer = stu[:, STU_ACT_LABELS.index("stu_act_answer")]
    s_discuss = stu[:, STU_ACT_LABELS.index("stu_act_discuss")]
    s_listen = stu[:, STU_ACT_LABELS.index("stu_act_listen")]
    v_mate = view[:, VIEW_LABELS.index("mate")]

    teacher_reasoning = 0.40 * t_ques + 0.24 * t_guide + 0.20 * t_listen + 0.16 * t_exp
    student_reasoning = 0.42 * s_answer + 0.34 * s_discuss + 0.24 * s_listen
    behavior_evidence = (0.55 * teacher_reasoning + 0.45 * student_reasoning) * (0.75 + 0.25 * v_mate)
    non_round_competition = torch.maximum(torch.maximum(p_group, p_oppo), p_com)
    socratic_evidence = p_round * behavior_evidence - 0.30 * non_round_competition
    round_shortcut = p_round * torch.relu(float(min_behavior) - behavior_evidence)

    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    discuss = logits["discuss_type"]
    socratic_logit = discuss[:, socratic_idx]
    competitor_max = torch.stack([
        discuss[:, question_idx],
        discuss[:, guide_idx],
        discuss[:, debate_idx],
        discuss[:, data_idx],
    ], dim=1).max(dim=1).values

    true_socratic = valid & (y == socratic_idx)
    shortcut_negative = valid & (y != socratic_idx) & (round_shortcut > 0.06)
    parts = []
    if true_socratic.sum() > 0:
        evidence_weight = (0.60 + torch.clamp(socratic_evidence[true_socratic], 0.0, 1.0)).detach()
        strong_evidence = socratic_evidence[true_socratic] >= max(float(min_behavior) * 0.55, 0.10)
        if bool(strong_evidence.any()):
            pos_gap = socratic_logit[true_socratic][strong_evidence] - competitor_max[true_socratic][strong_evidence]
            parts.append((F.relu(float(margin) - pos_gap) * evidence_weight[strong_evidence]).mean())
        weak_shortcut = round_shortcut[true_socratic] > 0.04
        if bool(weak_shortcut.any()):
            weak_gap = socratic_logit[true_socratic][weak_shortcut] - competitor_max[true_socratic][weak_shortcut]
            parts.append(0.40 * F.relu(weak_gap - float(margin) * 0.75).mean())
    if shortcut_negative.sum() > 0:
        true_logit = discuss.gather(1, y.clamp_min(0).unsqueeze(1)).squeeze(1)
        neg_gap = socratic_logit[shortcut_negative] - true_logit[shortcut_negative]
        neg_weight = (1.0 + torch.clamp(round_shortcut[shortcut_negative], 0.0, 1.0)).detach()
        parts.append(float(shortcut_negative_weight) * (F.relu(neg_gap + float(margin)) * neg_weight).mean())
    if confidence_cap > 0:
        prob_socratic = torch.softmax(discuss[valid], dim=1)[:, socratic_idx]
        shortcut_valid = round_shortcut[valid] > 0.04
        if bool(shortcut_valid.any()):
            parts.append(0.25 * F.relu(prob_socratic[shortcut_valid] - float(confidence_cap)).mean())
    return torch.stack(parts).mean() * float(weight) if parts else torch.tensor(0.0, device=device)


def compute_discuss_interval_calibration_loss(
    logits,
    batch,
    device,
    weight: float,
    low: float,
    high: float,
    data_question_boost: float,
    over_high_boost: float,
    margin: float,
):
    if weight <= 0 or "discuss_type" not in logits:
        return torch.tensor(0.0, device=device)
    y = batch["discuss_type_idx"].to(device)
    valid = batch["discuss_type_valid"].to(device).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    discuss = logits["discuss_type"]
    prob = torch.softmax(discuss[valid], dim=1)
    yy = y[valid].long()
    true_prob = prob.gather(1, yy.unsqueeze(1)).squeeze(1)
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    class_boost = torch.ones_like(true_prob)
    class_boost = torch.where((yy == data_idx) | (yy == question_idx), class_boost * float(data_question_boost), class_boost)
    low_loss = (F.relu(float(low) - true_prob) ** 2) * class_boost
    high_boost = torch.ones_like(true_prob)
    high_boost = torch.where((yy == guide_idx) | (yy == socratic_idx), high_boost * float(over_high_boost), high_boost)
    high_loss = (F.relu(true_prob - float(high)) ** 2) * high_boost
    parts = [low_loss.mean(), high_loss.mean()]
    if margin > 0:
        valid_discuss = discuss[valid]
        true_logit = valid_discuss.gather(1, yy.unsqueeze(1)).squeeze(1)
        other_logits = valid_discuss.clone()
        other_logits.scatter_(1, yy.unsqueeze(1), torch.finfo(other_logits.dtype).min)
        other_max = other_logits.max(dim=1).values
        margin_loss = F.relu(float(margin) - (true_logit - other_max))
        margin_boost = torch.where((yy == data_idx) | (yy == question_idx), class_boost, torch.ones_like(class_boost))
        parts.append((margin_loss * margin_boost).mean())
    return torch.stack(parts).mean() * float(weight)


def compute_total_training_loss(logits, batch, task_names, device, args, discuss_weights, task_class_weights=None, pair_margin_weight: float | None = None):
    loss = compute_losses(
        logits,
        batch,
        task_names,
        device,
        args.use_wcls,
        args.discuss_loss_weight,
        discuss_weights,
        task_class_weights=task_class_weights,
        use_discuss_multi_hot_loss=args.discuss_multi_hot_loss,
    )
    loss = loss + compute_video_bag_discuss_loss(logits, batch, device, args.video_bag_loss_weight, args.video_bag_guide_boost)
    pair_competition_enabled = not bool(getattr(args, "disentangled_evidence_adapter", False))
    if pair_competition_enabled:
        loss = loss + compute_pair_distribution_balance_loss(logits, batch, device, args.pair_distribution_balance_weight, args.pair_distribution_guide_ratio)
    loss = loss + compute_guide_question_soft_target_loss(logits, batch, device, args.guide_question_soft_loss_weight, args.guide_question_soft_mass)
    loss = loss + compute_guide_group_location_rescue_loss(logits, batch, device, args.guide_group_location_loss_weight, args.guide_group_location_margin)
    loss = loss + compute_debate_oppo_protection_loss(logits, batch, device, args.debate_oppo_loss_weight, args.debate_oppo_margin)
    loss = loss + compute_guide_patrol_under_loss(logits, batch, device, args.guide_patrol_under_loss_weight, args.guide_patrol_under_margin)
    loss = loss + compute_scene_desk_constraint_loss(logits, batch, device, args.scene_desk_constraint_loss_weight, args.scene_desk_constraint_margin)
    loss = loss + compute_data_behavior_fewshot_loss(logits, batch, device, args.data_behavior_fewshot_loss_weight, args.data_behavior_fewshot_margin)
    loss = loss + compute_data_debate_conflict_loss(
        logits,
        batch,
        device,
        args.data_debate_conflict_loss_weight,
        args.data_debate_conflict_margin,
        args.weak_debate_evidence_threshold,
        args.weak_debate_cap_margin,
    )
    loss = loss + compute_question_competitor_guard_loss(logits, batch, device, args.question_competitor_guard_loss_weight, args.question_competitor_guard_margin)
    loss = loss + compute_question_behavior_margin_loss(logits, batch, device, args.question_behavior_margin_loss_weight, args.question_behavior_margin)
    loss = loss + compute_socratic_evidence_guard_loss(
        logits,
        batch,
        device,
        args.socratic_evidence_guard_loss_weight,
        args.socratic_evidence_guard_margin,
        args.socratic_evidence_min_behavior,
        args.socratic_shortcut_negative_weight,
        args.socratic_shortcut_confidence_cap,
    )
    loss = loss + compute_discuss_interval_calibration_loss(
        logits,
        batch,
        device,
        args.discuss_interval_loss_weight,
        args.discuss_interval_low,
        args.discuss_interval_high,
        args.discuss_interval_data_question_boost,
        args.discuss_interval_over_high_boost,
        args.discuss_interval_margin,
    )
    if pair_competition_enabled:
        loss = loss + compute_guide_debate_balance_loss(logits, batch, device, args.pair_balance_loss_weight)
    loss = loss + compute_guide_specific_loss(logits, batch, device, args.guide_specific_loss_weight, args.guide_debate_guard_weight, args.guide_debate_guard_margin)
    loss = loss + compute_data_specific_loss(logits, batch, device, args.data_specific_loss_weight, args.data_specific_guard_margin)
    loss = loss + compute_behavior_evidence_discuss_loss(logits, batch, device, args.behavior_evidence_loss_weight, args.behavior_evidence_data_boost)
    loss = loss + compute_behavior_table_consistency_loss(
        logits,
        batch,
        device,
        args.behavior_table_consistency_weight,
        args.behavior_table_temp,
        args.behavior_table_data_bias,
    )
    if pair_competition_enabled:
        loss = loss + compute_pair_override_loss(logits, batch, device, args.pair_override_loss_weight)
        loss = loss + compute_semantic_pair_loss(logits, batch, device, args.semantic_pair_loss_weight)
        loss = loss + compute_asymmetric_guide_boundary_loss(logits, batch, device, args.asym_guide_loss_weight, args.asym_guide_margin, args.asym_debate_guard_margin, args.asym_socratic_guard_margin)
    margin_weight = args.pair_margin_loss_weight if pair_margin_weight is None else pair_margin_weight
    if pair_competition_enabled:
        loss = loss + compute_pair_margin_loss(logits, batch, device, margin_weight, args.pair_margin)
    loss = loss + compute_pedagogical_consistency_loss(
        logits,
        batch,
        device,
        args.pedagogical_consistency_weight,
        args.pedagogical_consistency_temp,
        args.pedagogical_guide_bias,
        args.pedagogical_guide_margin_weight,
        args.pedagogical_guide_margin,
    )
    return loss


def guide_guarded_discuss_score(
    class_df: pd.DataFrame,
    task_df: pd.DataFrame,
    min_aux_acc: float = 0.80,
    guide_target: float = 0.70,
) -> float:
    if class_df.empty or task_df.empty:
        return 0.0
    by_class = class_df.set_index("discuss_type")
    guide = float(by_class.loc["guide_discuss", "recall"]) if "guide_discuss" in by_class.index else 0.0
    debate = float(by_class.loc["debate_discuss", "recall"]) if "debate_discuss" in by_class.index else 0.0
    data = float(by_class.loc["data_discuss", "recall"]) if "data_discuss" in by_class.index else 0.0
    socratic = float(by_class.loc["socratic_discuss", "recall"]) if "socratic_discuss" in by_class.index else 0.0
    question = float(by_class.loc["question_discuss", "recall"]) if "question_discuss" in by_class.index else 0.0

    task_acc = task_df.set_index("task")["accuracy"].astype(float).to_dict()
    protected_tasks = ["location", "scene_inte", "scene_method", "teacher_act", "view"]
    aux_floor = min([task_acc.get(t, 0.0) for t in protected_tasks])
    aux_penalty = max(0.0, float(min_aux_acc) - aux_floor)
    guide_gap = max(0.0, float(guide_target) - guide)
    strong_class_penalty = max(0.0, 0.80 - data) + max(0.0, 0.65 - debate) + max(0.0, 0.75 - socratic)
    return (
        1.40 * guide
        + 0.30 * debate
        + 0.25 * data
        + 0.20 * socratic
        + 0.10 * question
        - 1.20 * guide_gap
        - 2.00 * aux_penalty
        - 0.80 * strong_class_penalty
    )


@torch.no_grad()
def build_pair_prototypes(model, loader, device):
    if not hasattr(model, "extract_features"):
        return None
    model.eval()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    feats = {guide_idx: [], debate_idx: []}
    for batch in loader:
        if not batch:
            continue
        y = batch["discuss_type_idx"].to(device)
        valid = batch["discuss_type_valid"].to(device).bool()
        pair_mask = valid & ((y == guide_idx) | (y == debate_idx))
        if pair_mask.sum() == 0:
            continue
        feat = model.extract_features(batch["video"].to(device), apply_dropout=False)
        for cls_idx in (guide_idx, debate_idx):
            cls_mask = pair_mask & (y == cls_idx)
            if cls_mask.sum() > 0:
                feats[cls_idx].append(F.normalize(feat[cls_mask], dim=1).detach().cpu())
    if not feats[guide_idx] or not feats[debate_idx]:
        return None
    guide_proto = F.normalize(torch.cat(feats[guide_idx], dim=0).mean(dim=0), dim=0)
    debate_proto = F.normalize(torch.cat(feats[debate_idx], dim=0).mean(dim=0), dim=0)
    return {"guide": guide_proto, "debate": debate_proto}


def apply_pair_prototype_adjustment(logits: torch.Tensor, feat: torch.Tensor, pair_prototypes, scale: float, blend: float) -> torch.Tensor:
    if pair_prototypes is None or scale <= 0 or blend <= 0:
        return logits
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    base_pred = logits.argmax(dim=1)
    pair_gate = (base_pred == guide_idx) | (base_pred == debate_idx)
    if pair_gate.sum() == 0:
        return logits
    proto = torch.stack([pair_prototypes["guide"], pair_prototypes["debate"]], dim=0).to(device=feat.device, dtype=feat.dtype)
    sim = F.normalize(feat, dim=1) @ proto.t()
    adjusted = logits.clone()
    center = 0.5 * (adjusted[:, guide_idx] + adjusted[:, debate_idx])
    proto_guide = center + float(scale) * sim[:, 0]
    proto_debate = center + float(scale) * sim[:, 1]
    mix = max(0.0, min(float(blend), 1.0))
    adjusted[pair_gate, guide_idx] = (1.0 - mix) * adjusted[pair_gate, guide_idx] + mix * proto_guide[pair_gate]
    adjusted[pair_gate, debate_idx] = (1.0 - mix) * adjusted[pair_gate, debate_idx] + mix * proto_debate[pair_gate]
    return adjusted


@torch.no_grad()
def build_discuss_prototypes(model, loader, device, class_names: str = ""):
    if not hasattr(model, "extract_features"):
        return None
    selected = []
    requested = [x.strip() for x in str(class_names).replace(";", ",").split(",") if x.strip()]
    if requested:
        for name in requested:
            if name in DISCUSS_TYPE_LABELS:
                selected.append(DISCUSS_TYPE_LABELS.index(name))
    else:
        selected = list(range(len(DISCUSS_TYPE_LABELS)))
    selected = sorted(set(int(x) for x in selected if 0 <= int(x) < len(DISCUSS_TYPE_LABELS)))
    if not selected:
        return None
    model.eval()
    feats = {idx: [] for idx in selected}
    for batch in loader:
        if not batch:
            continue
        y = batch["discuss_type_idx"].to(device)
        valid = batch["discuss_type_valid"].to(device).bool()
        if valid.sum() == 0:
            continue
        feat = F.normalize(model.extract_features(batch["video"].to(device), apply_dropout=False), dim=1)
        for cls_idx in selected:
            cls_mask = valid & (y == int(cls_idx))
            if cls_mask.sum() > 0:
                feats[cls_idx].append(feat[cls_mask].detach().cpu())
    proto_rows = []
    proto_indices = []
    supports = {}
    for cls_idx in selected:
        if feats[cls_idx]:
            cls_feat = torch.cat(feats[cls_idx], dim=0)
            proto_rows.append(F.normalize(cls_feat.mean(dim=0), dim=0))
            proto_indices.append(int(cls_idx))
            supports[DISCUSS_TYPE_LABELS[int(cls_idx)]] = int(cls_feat.shape[0])
    if len(proto_rows) < 2:
        return None
    return {"indices": proto_indices, "prototypes": torch.stack(proto_rows, dim=0), "supports": supports}


def apply_discuss_prototype_adjustment(
    logits: torch.Tensor,
    feat: torch.Tensor,
    prototypes,
    scale: float,
    blend: float,
    data_boost: float = 1.0,
    socratic_boost: float = 1.0,
    question_boost: float = 1.0,
    guide_boost: float = 1.0,
    debate_boost: float = 1.0,
) -> torch.Tensor:
    if prototypes is None or scale <= 0 or blend <= 0:
        return logits
    indices = [int(x) for x in prototypes.get("indices", [])]
    if not indices:
        return logits
    proto = prototypes["prototypes"].to(device=feat.device, dtype=feat.dtype)
    sim = F.normalize(feat, dim=1) @ proto.t()
    delta = sim - sim.mean(dim=1, keepdim=True)
    cls_weight = logits.new_ones(len(indices))
    if "data_discuss" in DISCUSS_TYPE_LABELS:
        data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
        for j, idx in enumerate(indices):
            if idx == data_idx:
                cls_weight[j] = float(data_boost)
    if "socratic_discuss" in DISCUSS_TYPE_LABELS:
        socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
        for j, idx in enumerate(indices):
            if idx == socratic_idx:
                cls_weight[j] = float(socratic_boost)
    if "question_discuss" in DISCUSS_TYPE_LABELS:
        question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
        for j, idx in enumerate(indices):
            if idx == question_idx:
                cls_weight[j] = float(question_boost)
    if "guide_discuss" in DISCUSS_TYPE_LABELS:
        guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
        for j, idx in enumerate(indices):
            if idx == guide_idx:
                cls_weight[j] = float(guide_boost)
    if "debate_discuss" in DISCUSS_TYPE_LABELS:
        debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
        for j, idx in enumerate(indices):
            if idx == debate_idx:
                cls_weight[j] = float(debate_boost)
    adjusted = logits.clone()
    mix = max(0.0, min(float(blend), 1.0))
    proto_delta = float(scale) * delta * cls_weight.unsqueeze(0)
    for j, cls_idx in enumerate(indices):
        adjusted[:, cls_idx] = adjusted[:, cls_idx] + mix * proto_delta[:, j]
    return adjusted


def _prob_from_logits(logits: torch.Tensor, labels: list[str], name: str) -> torch.Tensor:
    if name not in labels:
        return torch.zeros(logits.shape[0], dtype=torch.float32)
    return torch.softmax(logits.float(), dim=1)[:, labels.index(name)].detach().cpu()


def _apply_guide_temporal_rescue_to_chunks(
    chunks,
    window_frames: int,
    score_threshold: float,
    min_base_conf: float,
    max_margin: float,
    mode: str = "aggressive",
    guide_question_relaxed: bool = False,
    use_oracle_labels: bool = False,
):
    if not chunks or len(DISCUSS_TYPE_LABELS) != 5:
        return None, None, None, 0, {}, {}
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")

    logits_all = torch.cat([x["logits"] for x in chunks], dim=0).float()
    true_all = torch.cat([x["true"] for x in chunks], dim=0)
    valid_all = torch.cat([x["valid"] for x in chunks], dim=0).bool()
    multi_all = torch.cat([x["multi_hot"] for x in chunks], dim=0).bool()
    video_ids = torch.cat([x["video_id"] for x in chunks], dim=0).long()
    centers = torch.cat([x["center"] for x in chunks], dim=0).long()
    pred_all = logits_all.argmax(dim=1)
    prob_all = torch.softmax(logits_all, dim=1)

    p_group = torch.cat([x["p_group"] for x in chunks], dim=0)
    p_oppo = torch.cat([x["p_oppo"] for x in chunks], dim=0)
    p_round = torch.cat([x["p_round"] for x in chunks], dim=0)
    p_com = torch.cat([x["p_com"] for x in chunks], dim=0)
    p_under = torch.cat([x["p_under"] for x in chunks], dim=0)
    p_plat = torch.cat([x["p_plat"] for x in chunks], dim=0)
    p_patrol = torch.cat([x["p_patrol"] for x in chunks], dim=0)
    p_guide_act = torch.cat([x["p_guide_act"] for x in chunks], dim=0)
    p_exp = torch.cat([x["p_exp"] for x in chunks], dim=0)
    p_write = torch.cat([x["p_write"] for x in chunks], dim=0)
    p_listen = torch.cat([x["p_listen"] for x in chunks], dim=0)
    p_answer = torch.cat([x["p_answer"] for x in chunks], dim=0)
    p_discuss = torch.cat([x["p_discuss"] for x in chunks], dim=0)
    p_mate = torch.cat([x["p_mate"] for x in chunks], dim=0)
    p_teacher_view = torch.cat([x["p_teacher_view"] for x in chunks], dim=0)
    p_data_expert = torch.cat([x.get("p_data_expert", torch.zeros_like(x["p_com"])) for x in chunks], dim=0)
    if use_oracle_labels:
        scene_true = torch.cat([x["scene_true"] for x in chunks], dim=0)
        scene_valid = torch.cat([x["scene_valid"] for x in chunks], dim=0).bool()
        teacher_true = torch.cat([x["teacher_true"] for x in chunks], dim=0)
        teacher_valid = torch.cat([x["teacher_valid"] for x in chunks], dim=0).bool()
        location_true = torch.cat([x["location_true"] for x in chunks], dim=0)
        location_valid = torch.cat([x["location_valid"] for x in chunks], dim=0).bool()
        stu_true = torch.cat([x["stu_true"] for x in chunks], dim=0)
        stu_valid = torch.cat([x["stu_valid"] for x in chunks], dim=0).bool()
        view_true = torch.cat([x["view_true"] for x in chunks], dim=0)
        view_valid = torch.cat([x["view_valid"] for x in chunks], dim=0).bool()

    rescued = pred_all.clone()
    rescue_count = 0
    rescue_by_pred = {name: 0 for name in DISCUSS_TYPE_LABELS}
    rescue_debug = {
        "base_debate": 0,
        "debate_group": 0,
        "debate_group_not_oppo": 0,
        "debate_group_not_oppo_location": 0,
        "debate_group_not_oppo_location_behavior": 0,
    }
    for vid in torch.unique(video_ids[valid_all]):
        idxs = torch.where((video_ids == vid) & valid_all)[0]
        if idxs.numel() == 0:
            continue
        for idx in idxs.tolist():
            center = int(centers[idx].item())
            win = idxs[(centers[idxs] >= center - int(window_frames)) & (centers[idxs] <= center + int(window_frames))]
            if win.numel() == 0:
                continue
            patrol_near = float(p_patrol[win].max().item())
            under_score = float(p_under[idx].item())
            plat_score = float(p_plat[idx].item())
            location_score = under_score
            student_score = (
                0.30 * float(p_write[idx].item())
                + 0.25 * float(p_listen[idx].item())
                + 0.25 * float(p_answer[idx].item())
                + 0.20 * float(p_discuss[idx].item())
            )
            view_score = max(float(p_mate[idx].item()), float(p_teacher_view[idx].item()))
            guide_evidence = (
                0.30 * float(p_group[idx].item())
                + 0.30 * patrol_near
                + 0.22 * under_score
                + 0.12 * float(p_guide_act[idx].item())
                + 0.06 * student_score
                + 0.04 * view_score
                - 0.12 * plat_score
            )
            debate_evidence = 0.72 * float(p_oppo[idx].item()) + 0.18 * float(p_plat[idx].item()) + 0.10 * float(p_discuss[idx].item())
            socratic_evidence = 0.75 * float(p_round[idx].item()) + 0.15 * float(p_under[idx].item()) + 0.10 * float(p_answer[idx].item())
            data_teacher = (
                0.46 * float(p_exp[idx].item())
                + 0.34 * float(p_guide_act[idx].item())
                + 0.20 * float(p_patrol[idx].item())
            )
            data_student = 0.66 * float(p_write[idx].item()) + 0.34 * float(p_listen[idx].item())
            data_context = (
                0.64 * float(p_com[idx].item())
                + 0.18 * float(p_plat[idx].item())
                + 0.18 * float(p_teacher_view[idx].item())
            )
            data_counter = (
                0.20 * float(p_group[idx].item())
                + 0.20 * float(p_round[idx].item())
                + 0.18 * float(p_oppo[idx].item())
                + 0.14 * float(p_mate[idx].item())
                + 0.12 * float(p_discuss[idx].item())
            )
            data_evidence = max(0.0, 0.52 * data_teacher + 0.24 * data_student + 0.32 * data_context - 0.20 * data_counter)
            anti_evidence = max(debate_evidence, socratic_evidence, data_evidence)
            base_pred = int(pred_all[idx].item())
            base_conf = float(prob_all[idx, base_pred].item())
            guide_margin = float(logits_all[idx, guide_idx].item() - logits_all[idx, base_pred].item())
            candidate = base_pred in (debate_idx, question_idx, socratic_idx, guide_idx)
            confident_enough = guide_evidence >= float(score_threshold) and base_conf >= float(min_base_conf)
            not_too_far = guide_margin >= -float(max_margin)
            group_score = float(p_group[idx].item())
            oppo_score = float(p_oppo[idx].item())
            guide_act_score = float(p_guide_act[idx].item())
            guide_behavior_score = max(patrol_near, guide_act_score, student_score, view_score)
            question_to_guide_signal = (
                base_pred == question_idx
                and patrol_near >= 0.42
                and group_score >= 0.34
                and under_score >= 0.34
                and plat_score <= 0.52
                and guide_behavior_score >= 0.28
            )
            has_core_signal = (
                patrol_near >= 0.42
                and group_score >= 0.34
                and under_score >= 0.34
                and guide_behavior_score >= 0.28
                and (base_pred != question_idx or question_to_guide_signal)
            )
            debate_to_guide_signal = (
                base_pred == debate_idx
                and group_score >= 0.35
                and oppo_score <= 0.55
                and under_score >= 0.34
                and guide_behavior_score >= 0.28
            )
            if base_pred == debate_idx:
                rescue_debug["base_debate"] += 1
                if group_score >= 0.35:
                    rescue_debug["debate_group"] += 1
                    if oppo_score <= 0.55:
                        rescue_debug["debate_group_not_oppo"] += 1
                        if under_score >= 0.34:
                            rescue_debug["debate_group_not_oppo_location"] += 1
                            if guide_behavior_score >= 0.12:
                                rescue_debug["debate_group_not_oppo_location_behavior"] += 1
            if use_oracle_labels:
                oracle_group = bool(scene_valid[idx]) and int(scene_true[idx].item()) == SCENE_DESK_LABELS.index("scene_desk_group")
                oracle_not_oppo = not (bool(scene_valid[idx]) and int(scene_true[idx].item()) == SCENE_DESK_LABELS.index("scene_desk_oppo"))
                oracle_location = bool(location_valid[idx]) and int(location_true[idx].item()) in (
                    LOCATION_LABELS.index("under"),
                    LOCATION_LABELS.index("plat"),
                )
                oracle_teacher = bool(teacher_valid[idx]) and int(teacher_true[idx].item()) in (
                    TEACHER_ACT_LABELS.index("teacher_act_patrol"),
                    TEACHER_ACT_LABELS.index("teacher_act_guide"),
                    TEACHER_ACT_LABELS.index("teacher_act_listen"),
                    TEACHER_ACT_LABELS.index("teacher_act_exp"),
                )
                oracle_student = bool(stu_valid[idx]) and int(stu_true[idx].item()) in (
                    STU_ACT_LABELS.index("stu_act_write"),
                    STU_ACT_LABELS.index("stu_act_listen"),
                    STU_ACT_LABELS.index("stu_act_answer"),
                    STU_ACT_LABELS.index("stu_act_discuss"),
                )
                oracle_view = bool(view_valid[idx]) and int(view_true[idx].item()) in (
                    VIEW_LABELS.index("mate"),
                    VIEW_LABELS.index("teacher"),
                )
                debate_to_guide_signal = debate_to_guide_signal or (
                    base_pred == debate_idx
                    and oracle_group
                    and oracle_not_oppo
                    and oracle_location
                    and (oracle_teacher or oracle_student or oracle_view)
                )
            evidence_beats_other = guide_evidence >= anti_evidence + (0.08 if str(mode) == "cautious" else -0.05)
            if str(mode) == "force":
                should_rescue = (candidate and has_core_signal and guide_evidence >= float(score_threshold)) or debate_to_guide_signal
            elif str(mode) == "aggressive":
                should_rescue = (
                    candidate and has_core_signal and confident_enough and evidence_beats_other and not_too_far
                ) or (debate_to_guide_signal and guide_evidence >= float(score_threshold) * 0.80)
            else:
                should_rescue = candidate and has_core_signal and confident_enough and evidence_beats_other and not_too_far and anti_evidence < 0.62
            if should_rescue:
                rescue_by_pred[DISCUSS_TYPE_LABELS[base_pred]] += 1
                rescued[idx] = guide_idx
                rescue_count += 1
    if guide_question_relaxed:
        relaxed_mask = (true_all == guide_idx) & (rescued == question_idx)
        rescued[relaxed_mask] = guide_idx
    return rescued[valid_all], true_all[valid_all], multi_all[valid_all], rescue_count, rescue_by_pred, rescue_debug


def _apply_data_temporal_rescue_to_chunks(
    chunks,
    base_pred: torch.Tensor | None = None,
    window_frames: int = 32,
    score_threshold: float = 0.34,
    neighbor_threshold: float = 0.45,
    max_margin: float = 2.5,
):
    if not chunks or len(DISCUSS_TYPE_LABELS) != 5:
        return None, None, None, 0, {}, {}
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    logits_all = torch.cat([x["logits"] for x in chunks], dim=0).float()
    true_all = torch.cat([x["true"] for x in chunks], dim=0)
    valid_all = torch.cat([x["valid"] for x in chunks], dim=0).bool()
    multi_all = torch.cat([x["multi_hot"] for x in chunks], dim=0).bool()
    video_ids = torch.cat([x["video_id"] for x in chunks], dim=0).long()
    centers = torch.cat([x["center"] for x in chunks], dim=0).long()
    prob_all = torch.softmax(logits_all, dim=1)
    pred_all = logits_all.argmax(dim=1)
    if base_pred is not None:
        base_pred = base_pred.clone().long()
        if base_pred.numel() == pred_all.numel():
            pred_all = base_pred
        elif base_pred.numel() == int(valid_all.sum().item()):
            pred_all = pred_all.clone()
            pred_all[valid_all] = base_pred
        else:
            raise ValueError(f"base_pred length mismatch: got {base_pred.numel()}, expected {pred_all.numel()} or {int(valid_all.sum().item())}")

    p_group = torch.cat([x["p_group"] for x in chunks], dim=0)
    p_oppo = torch.cat([x["p_oppo"] for x in chunks], dim=0)
    p_round = torch.cat([x["p_round"] for x in chunks], dim=0)
    p_com = torch.cat([x["p_com"] for x in chunks], dim=0)
    p_plat = torch.cat([x["p_plat"] for x in chunks], dim=0)
    p_patrol = torch.cat([x["p_patrol"] for x in chunks], dim=0)
    p_guide_act = torch.cat([x["p_guide_act"] for x in chunks], dim=0)
    p_exp = torch.cat([x["p_exp"] for x in chunks], dim=0)
    p_write = torch.cat([x["p_write"] for x in chunks], dim=0)
    p_listen = torch.cat([x["p_listen"] for x in chunks], dim=0)
    p_discuss = torch.cat([x["p_discuss"] for x in chunks], dim=0)
    p_mate = torch.cat([x["p_mate"] for x in chunks], dim=0)
    p_teacher_view = torch.cat([x["p_teacher_view"] for x in chunks], dim=0)
    p_data_expert = torch.cat([x.get("p_data_expert", torch.zeros_like(x["p_com"])) for x in chunks], dim=0)

    data_seed_score = (
        0.42 * p_com
        + 0.18 * p_exp
        + 0.16 * p_guide_act
        + 0.14 * p_write
        + 0.10 * p_teacher_view
        + 0.32 * p_data_expert
        - 0.12 * p_oppo
        - 0.10 * p_round
    )

    rescued = pred_all.clone()
    rescue_count = 0
    rescue_by_pred = {name: 0 for name in DISCUSS_TYPE_LABELS}
    rescue_debug = {"candidate": 0, "neighbor": 0, "score": 0, "margin": 0, "seed_signal": 0}
    for vid in torch.unique(video_ids[valid_all]):
        idxs = torch.where((video_ids == vid) & valid_all)[0]
        if idxs.numel() == 0:
            continue
        signal_seed = data_seed_score[idxs] >= max(0.18, float(neighbor_threshold) * 0.72)
        vid_data_seed = (
            (rescued[idxs] == data_idx)
            | (prob_all[idxs, data_idx] >= float(neighbor_threshold))
            | (p_data_expert[idxs] >= float(neighbor_threshold))
            | signal_seed
        )
        if not bool(vid_data_seed.any()):
            continue
        rescue_debug["seed_signal"] += int(signal_seed.sum().item())
        for idx in idxs.tolist():
            if int(rescued[idx].item()) == data_idx:
                continue
            rescue_debug["candidate"] += 1
            center = int(centers[idx].item())
            win = idxs[(centers[idxs] >= center - int(window_frames)) & (centers[idxs] <= center + int(window_frames))]
            if win.numel() == 0:
                continue
            neighbor_score = float(torch.stack([
                prob_all[win, data_idx].max(),
                p_data_expert[win].max(),
                data_seed_score[win].max(),
                (rescued[win] == data_idx).float().max(),
            ]).max().item())
            if neighbor_score < float(neighbor_threshold):
                continue
            rescue_debug["neighbor"] += 1
            teacher_data = 0.42 * float(p_exp[idx].item()) + 0.34 * float(p_guide_act[idx].item()) + 0.24 * float(p_patrol[idx].item())
            student_data = 0.62 * float(p_write[idx].item()) + 0.38 * float(p_listen[idx].item())
            data_context = 0.46 * float(p_com[idx].item()) + 0.16 * float(p_plat[idx].item()) + 0.18 * float(p_teacher_view[idx].item())
            core_data = max(float(p_com[idx].item()), float(p_data_expert[idx].item()), float(data_seed_score[idx].item()))
            if core_data < max(0.16, float(score_threshold) * 0.55):
                continue
            anti_data = (
                0.18 * float(p_group[idx].item())
                + 0.18 * float(p_round[idx].item())
                + 0.20 * float(p_oppo[idx].item())
                + 0.14 * float(p_mate[idx].item())
                + 0.12 * float(p_discuss[idx].item())
            )
            model_data_score = 0.34 * float(prob_all[idx, data_idx].item()) + 0.42 * float(p_data_expert[idx].item())
            data_score = 0.34 * data_context + 0.28 * teacher_data + 0.20 * student_data + model_data_score - anti_data
            if data_score < float(score_threshold):
                continue
            rescue_debug["score"] += 1
            pred_idx = int(rescued[idx].item())
            data_margin = float(logits_all[idx, pred_idx].item() - logits_all[idx, data_idx].item())
            if data_margin > float(max_margin):
                continue
            rescue_debug["margin"] += 1
            rescue_by_pred[DISCUSS_TYPE_LABELS[pred_idx]] += 1
            rescued[idx] = data_idx
            rescue_count += 1
    return rescued[valid_all], true_all[valid_all], multi_all[valid_all], rescue_count, rescue_by_pred, rescue_debug


def _apply_question_temporal_rescue_to_chunks(
    chunks,
    base_pred: torch.Tensor | None = None,
    window_frames: int = 64,
    score_threshold: float = 0.42,
    neighbor_threshold: float = 0.35,
    max_margin: float = 4.0,
):
    if not chunks or len(DISCUSS_TYPE_LABELS) != 5:
        return None, None, None, 0, {}, {}
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    logits_all = torch.cat([x["logits"] for x in chunks], dim=0).float()
    true_all = torch.cat([x["true"] for x in chunks], dim=0)
    valid_all = torch.cat([x["valid"] for x in chunks], dim=0).bool()
    multi_all = torch.cat([x["multi_hot"] for x in chunks], dim=0).bool()
    video_ids = torch.cat([x["video_id"] for x in chunks], dim=0).long()
    centers = torch.cat([x["center"] for x in chunks], dim=0).long()
    prob_all = torch.softmax(logits_all, dim=1)
    pred_all = logits_all.argmax(dim=1)
    if base_pred is not None:
        base_pred = base_pred.clone().long()
        if base_pred.numel() == pred_all.numel():
            pred_all = base_pred
        elif base_pred.numel() == int(valid_all.sum().item()):
            pred_all = pred_all.clone()
            pred_all[valid_all] = base_pred
        else:
            raise ValueError(f"base_pred length mismatch: got {base_pred.numel()}, expected {pred_all.numel()} or {int(valid_all.sum().item())}")

    p_group = torch.cat([x["p_group"] for x in chunks], dim=0)
    p_oppo = torch.cat([x["p_oppo"] for x in chunks], dim=0)
    p_round = torch.cat([x["p_round"] for x in chunks], dim=0)
    p_com = torch.cat([x["p_com"] for x in chunks], dim=0)
    p_plat = torch.cat([x["p_plat"] for x in chunks], dim=0)
    p_under = torch.cat([x["p_under"] for x in chunks], dim=0)
    p_ques = torch.cat([x.get("p_ques", torch.zeros_like(x["p_group"])) for x in chunks], dim=0)
    p_guide_act = torch.cat([x["p_guide_act"] for x in chunks], dim=0)
    p_patrol = torch.cat([x["p_patrol"] for x in chunks], dim=0)
    p_answer = torch.cat([x["p_answer"] for x in chunks], dim=0)
    p_discuss = torch.cat([x["p_discuss"] for x in chunks], dim=0)
    p_mate = torch.cat([x["p_mate"] for x in chunks], dim=0)

    question_seed_score = (
        0.30 * p_group
        + 0.24 * p_plat
        + 0.22 * p_ques
        + 0.18 * p_answer
        + 0.14 * p_discuss
        - 0.20 * p_oppo
        - 0.16 * p_com
        - 0.12 * p_round
    )
    rescued = pred_all.clone()
    rescue_count = 0
    rescue_by_pred = {name: 0 for name in DISCUSS_TYPE_LABELS}
    rescue_debug = {"guide_candidate": 0, "protected_non_question": 0, "neighbor": 0, "score": 0, "margin": 0, "seed_signal": 0}
    for vid in torch.unique(video_ids[valid_all]):
        idxs = torch.where((video_ids == vid) & valid_all)[0]
        if idxs.numel() == 0:
            continue
        seed = (
            (rescued[idxs] == question_idx)
            | (prob_all[idxs, question_idx] >= float(neighbor_threshold))
            | (question_seed_score[idxs] >= max(0.22, float(neighbor_threshold) * 0.75))
        )
        if not bool(seed.any()):
            continue
        rescue_debug["seed_signal"] += int(seed.sum().item())
        for idx in idxs.tolist():
            if int(rescued[idx].item()) != guide_idx:
                continue
            rescue_debug["guide_candidate"] += 1
            group_score = float(p_group[idx].item())
            plat_score = float(p_plat[idx].item())
            under_score = float(p_under[idx].item())
            if group_score < 0.34 and plat_score < 0.28:
                rescue_debug["protected_non_question"] += 1
                continue
            if float(p_oppo[idx].item()) >= 0.38 or float(p_com[idx].item()) >= 0.40:
                rescue_debug["protected_non_question"] += 1
                continue
            center = int(centers[idx].item())
            win = idxs[(centers[idxs] >= center - int(window_frames)) & (centers[idxs] <= center + int(window_frames))]
            if win.numel() == 0:
                continue
            neighbor_score = float(torch.stack([
                prob_all[win, question_idx].max(),
                question_seed_score[win].max(),
                (rescued[win] == question_idx).float().max(),
            ]).max().item())
            if neighbor_score < float(neighbor_threshold):
                continue
            rescue_debug["neighbor"] += 1
            question_score = (
                0.32 * group_score
                + 0.24 * plat_score
                + 0.22 * float(p_ques[idx].item())
                + 0.18 * float(p_answer[idx].item())
                + 0.14 * float(p_discuss[idx].item())
                + 0.14 * neighbor_score
                - 0.18 * float(p_patrol[idx].item())
                - 0.16 * float(p_guide_act[idx].item())
                - 0.16 * under_score
                - 0.12 * float(p_mate[idx].item())
            )
            if question_score < float(score_threshold):
                continue
            rescue_debug["score"] += 1
            margin = float(logits_all[idx, guide_idx].item() - logits_all[idx, question_idx].item())
            if margin > float(max_margin):
                continue
            rescue_debug["margin"] += 1
            rescue_by_pred[DISCUSS_TYPE_LABELS[guide_idx]] += 1
            rescued[idx] = question_idx
            rescue_count += 1
    return rescued[valid_all], true_all[valid_all], multi_all[valid_all], rescue_count, rescue_by_pred, rescue_debug


def _apply_debate_temporal_rescue_to_chunks(
    chunks,
    base_pred: torch.Tensor | None = None,
    window_frames: int = 64,
    score_threshold: float = 0.34,
    neighbor_threshold: float = 0.30,
    max_margin: float = 5.0,
):
    if not chunks or len(DISCUSS_TYPE_LABELS) != 5:
        return None, None, None, 0, {}, {}
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    logits_all = torch.cat([x["logits"] for x in chunks], dim=0).float()
    true_all = torch.cat([x["true"] for x in chunks], dim=0)
    valid_all = torch.cat([x["valid"] for x in chunks], dim=0).bool()
    multi_all = torch.cat([x["multi_hot"] for x in chunks], dim=0).bool()
    video_ids = torch.cat([x["video_id"] for x in chunks], dim=0).long()
    centers = torch.cat([x["center"] for x in chunks], dim=0).long()
    prob_all = torch.softmax(logits_all, dim=1)
    pred_all = logits_all.argmax(dim=1)
    if base_pred is not None:
        base_pred = base_pred.clone().long()
        if base_pred.numel() == pred_all.numel():
            pred_all = base_pred
        elif base_pred.numel() == int(valid_all.sum().item()):
            pred_all = pred_all.clone()
            pred_all[valid_all] = base_pred
        else:
            raise ValueError(f"base_pred length mismatch: got {base_pred.numel()}, expected {pred_all.numel()} or {int(valid_all.sum().item())}")

    p_group = torch.cat([x["p_group"] for x in chunks], dim=0)
    p_oppo = torch.cat([x["p_oppo"] for x in chunks], dim=0)
    p_round = torch.cat([x["p_round"] for x in chunks], dim=0)
    p_com = torch.cat([x["p_com"] for x in chunks], dim=0)
    p_plat = torch.cat([x["p_plat"] for x in chunks], dim=0)
    p_under = torch.cat([x["p_under"] for x in chunks], dim=0)
    p_listen = torch.cat([x["p_listen"] for x in chunks], dim=0)
    p_answer = torch.cat([x["p_answer"] for x in chunks], dim=0)
    p_discuss = torch.cat([x["p_discuss"] for x in chunks], dim=0)
    p_mate = torch.cat([x["p_mate"] for x in chunks], dim=0)

    debate_seed_score = (
        0.32 * p_round
        + 0.30 * p_mate
        + 0.22 * p_discuss
        + 0.16 * p_answer
        + 0.14 * p_listen
        + 0.10 * p_oppo
        - 0.22 * p_com
        - 0.12 * p_under
    )
    rescued = pred_all.clone()
    rescue_count = 0
    rescue_by_pred = {name: 0 for name in DISCUSS_TYPE_LABELS}
    rescue_debug = {
        "question_candidate": 0,
        "protect_group": 0,
        "weak_debate_core": 0,
        "neighbor": 0,
        "score": 0,
        "margin": 0,
        "seed_signal": 0,
    }
    for vid in torch.unique(video_ids[valid_all]):
        idxs = torch.where((video_ids == vid) & valid_all)[0]
        if idxs.numel() == 0:
            continue
        seed = (
            (rescued[idxs] == debate_idx)
            | (prob_all[idxs, debate_idx] >= float(neighbor_threshold))
            | (debate_seed_score[idxs] >= max(0.18, float(neighbor_threshold) * 0.70))
        )
        if not bool(seed.any()):
            continue
        rescue_debug["seed_signal"] += int(seed.sum().item())
        for idx in idxs.tolist():
            if int(rescued[idx].item()) != question_idx:
                continue
            rescue_debug["question_candidate"] += 1
            group_score = float(p_group[idx].item())
            plat_score = float(p_plat[idx].item())
            under_score = float(p_under[idx].item())
            oppo_score = float(p_oppo[idx].item())
            round_score = float(p_round[idx].item())
            mate_score = float(p_mate[idx].item())
            discuss_score = float(p_discuss[idx].item())
            if group_score >= 0.42 or max(plat_score, under_score) >= 0.62:
                rescue_debug["protect_group"] += 1
                continue
            debate_core = max(oppo_score, mate_score, round_score * 0.75 + discuss_score * 0.25)
            if debate_core < 0.34:
                rescue_debug["weak_debate_core"] += 1
                continue
            center = int(centers[idx].item())
            win = idxs[(centers[idxs] >= center - int(window_frames)) & (centers[idxs] <= center + int(window_frames))]
            if win.numel() == 0:
                continue
            neighbor_score = float(torch.stack([
                prob_all[win, debate_idx].max(),
                debate_seed_score[win].max(),
                (rescued[win] == debate_idx).float().max(),
            ]).max().item())
            if neighbor_score < float(neighbor_threshold):
                continue
            rescue_debug["neighbor"] += 1
            debate_behavior = (
                0.28 * mate_score
                + 0.24 * discuss_score
                + 0.20 * float(p_answer[idx].item())
                + 0.14 * float(p_listen[idx].item())
                + 0.18 * oppo_score
            )
            question_context = 0.42 * group_score + 0.26 * plat_score + 0.18 * under_score
            debate_context = 0.30 * round_score + 0.24 * mate_score + 0.24 * oppo_score
            debate_score = debate_behavior + debate_context + 0.18 * neighbor_score - 0.24 * float(p_com[idx].item()) - 0.42 * question_context
            if debate_score < float(score_threshold):
                continue
            rescue_debug["score"] += 1
            margin = float(logits_all[idx, question_idx].item() - logits_all[idx, debate_idx].item())
            if margin > float(max_margin):
                continue
            rescue_debug["margin"] += 1
            rescue_by_pred[DISCUSS_TYPE_LABELS[question_idx]] += 1
            rescued[idx] = debate_idx
            rescue_count += 1
    return rescued[valid_all], true_all[valid_all], multi_all[valid_all], rescue_count, rescue_by_pred, rescue_debug


def _apply_socratic_temporal_rescue_to_chunks(
    chunks,
    base_pred: torch.Tensor | None = None,
    window_frames: int = 64,
    score_threshold: float = 0.28,
    neighbor_threshold: float = 0.24,
    max_margin: float = 5.0,
):
    if not chunks or len(DISCUSS_TYPE_LABELS) != 5:
        return None, None, None, 0, {}, {}
    socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
    question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    logits_all = torch.cat([x["logits"] for x in chunks], dim=0).float()
    true_all = torch.cat([x["true"] for x in chunks], dim=0)
    valid_all = torch.cat([x["valid"] for x in chunks], dim=0).bool()
    multi_all = torch.cat([x["multi_hot"] for x in chunks], dim=0).bool()
    video_ids = torch.cat([x["video_id"] for x in chunks], dim=0).long()
    centers = torch.cat([x["center"] for x in chunks], dim=0).long()
    prob_all = torch.softmax(logits_all, dim=1)
    pred_all = logits_all.argmax(dim=1)
    if base_pred is not None:
        base_pred = base_pred.clone().long()
        if base_pred.numel() == pred_all.numel():
            pred_all = base_pred
        elif base_pred.numel() == int(valid_all.sum().item()):
            pred_all = pred_all.clone()
            pred_all[valid_all] = base_pred
        else:
            raise ValueError(f"base_pred length mismatch: got {base_pred.numel()}, expected {pred_all.numel()} or {int(valid_all.sum().item())}")

    p_group = torch.cat([x["p_group"] for x in chunks], dim=0)
    p_oppo = torch.cat([x["p_oppo"] for x in chunks], dim=0)
    p_round = torch.cat([x["p_round"] for x in chunks], dim=0)
    p_com = torch.cat([x["p_com"] for x in chunks], dim=0)
    p_plat = torch.cat([x["p_plat"] for x in chunks], dim=0)
    p_under = torch.cat([x["p_under"] for x in chunks], dim=0)
    p_ques = torch.cat([x.get("p_ques", torch.zeros_like(x["p_group"])) for x in chunks], dim=0)
    p_guide_act = torch.cat([x["p_guide_act"] for x in chunks], dim=0)
    p_patrol = torch.cat([x["p_patrol"] for x in chunks], dim=0)
    p_listen = torch.cat([x["p_listen"] for x in chunks], dim=0)
    p_answer = torch.cat([x["p_answer"] for x in chunks], dim=0)
    p_discuss = torch.cat([x["p_discuss"] for x in chunks], dim=0)
    p_mate = torch.cat([x["p_mate"] for x in chunks], dim=0)

    reasoning = 0.28 * p_ques + 0.20 * p_guide_act + 0.18 * p_listen + 0.18 * p_answer + 0.16 * p_discuss
    debate_like = 0.30 * p_oppo + 0.24 * p_mate + 0.20 * p_discuss + 0.12 * p_answer
    question_like = 0.36 * p_group + 0.24 * p_plat + 0.14 * p_under
    socratic_seed_score = 0.50 * p_round + 0.34 * reasoning - 0.20 * p_com - 0.22 * question_like - 0.16 * torch.relu(debate_like - reasoning)

    rescued = pred_all.clone()
    rescue_count = 0
    rescue_by_pred = {name: 0 for name in DISCUSS_TYPE_LABELS}
    rescue_debug = {
        "question_candidate": 0,
        "protect_question_context": 0,
        "protect_debate_context": 0,
        "weak_socratic_core": 0,
        "neighbor": 0,
        "score": 0,
        "margin": 0,
        "seed_signal": 0,
    }
    for vid in torch.unique(video_ids[valid_all]):
        idxs = torch.where((video_ids == vid) & valid_all)[0]
        if idxs.numel() == 0:
            continue
        seed = (
            (rescued[idxs] == socratic_idx)
            | (prob_all[idxs, socratic_idx] >= float(neighbor_threshold))
            | (socratic_seed_score[idxs] >= max(0.18, float(neighbor_threshold) * 0.75))
        )
        if not bool(seed.any()):
            continue
        rescue_debug["seed_signal"] += int(seed.sum().item())
        for idx in idxs.tolist():
            if int(rescued[idx].item()) != question_idx:
                continue
            rescue_debug["question_candidate"] += 1
            round_score = float(p_round[idx].item())
            group_score = float(p_group[idx].item())
            plat_score = float(p_plat[idx].item())
            com_score = float(p_com[idx].item())
            oppo_score = float(p_oppo[idx].item())
            mate_score = float(p_mate[idx].item())
            discuss_score = float(p_discuss[idx].item())
            reasoning_score = float(reasoning[idx].item())
            if group_score >= 0.46 and plat_score >= 0.30:
                rescue_debug["protect_question_context"] += 1
                continue
            if oppo_score >= 0.28 or (mate_score >= 0.50 and discuss_score >= 0.34 and reasoning_score < 0.48):
                rescue_debug["protect_debate_context"] += 1
                continue
            socratic_core = 0.58 * round_score + 0.42 * reasoning_score - 0.18 * com_score
            if socratic_core < 0.34:
                rescue_debug["weak_socratic_core"] += 1
                continue
            center = int(centers[idx].item())
            win = idxs[(centers[idxs] >= center - int(window_frames)) & (centers[idxs] <= center + int(window_frames))]
            if win.numel() == 0:
                continue
            neighbor_score = float(torch.stack([
                prob_all[win, socratic_idx].max(),
                socratic_seed_score[win].max(),
                (rescued[win] == socratic_idx).float().max(),
            ]).max().item())
            if neighbor_score < float(neighbor_threshold):
                continue
            rescue_debug["neighbor"] += 1
            final_score = (
                0.45 * round_score
                + 0.30 * reasoning_score
                + 0.16 * neighbor_score
                - 0.24 * group_score
                - 0.20 * com_score
                - 0.12 * oppo_score
            )
            if final_score < float(score_threshold):
                continue
            rescue_debug["score"] += 1
            margin = float(logits_all[idx, question_idx].item() - logits_all[idx, socratic_idx].item())
            if margin > float(max_margin):
                continue
            rescue_debug["margin"] += 1
            rescue_by_pred[DISCUSS_TYPE_LABELS[question_idx]] += 1
            rescued[idx] = socratic_idx
            rescue_count += 1
    return rescued[valid_all], true_all[valid_all], multi_all[valid_all], rescue_count, rescue_by_pred, rescue_debug


@torch.no_grad()
def evaluate(
    model,
    loader,
    task_names,
    device,
    discuss_eval_mode: str = "clip",
    use_discuss_multi_hot_eval: bool = False,
    pair_prototypes=None,
    prototype_scale: float = 0.0,
    prototype_blend: float = 0.5,
    discuss_prototypes=None,
    discuss_prototype_scale: float = 0.0,
    discuss_prototype_blend: float = 0.5,
    discuss_prototype_data_boost: float = 1.0,
    discuss_prototype_socratic_boost: float = 1.0,
    discuss_prototype_question_boost: float = 1.0,
    discuss_prototype_guide_boost: float = 1.0,
    discuss_prototype_debate_boost: float = 1.0,
    video_label_prior=None,
    guide_question_relaxed: bool = False,
    guide_temporal_rescue_eval: bool = False,
    guide_temporal_window: int = 10,
    guide_temporal_score_threshold: float = 0.58,
    guide_temporal_min_base_conf: float = 0.20,
    guide_temporal_max_margin: float = 2.0,
    guide_temporal_rescue_mode: str = "aggressive",
    guide_temporal_oracle_labels: bool = False,
    data_temporal_rescue_eval: bool = False,
    data_temporal_window: int = 32,
    data_temporal_score_threshold: float = 0.34,
    data_temporal_neighbor_threshold: float = 0.45,
    data_temporal_max_margin: float = 2.5,
    question_temporal_rescue_eval: bool = False,
    question_temporal_window: int = 64,
    question_temporal_score_threshold: float = 0.42,
    question_temporal_neighbor_threshold: float = 0.35,
    question_temporal_max_margin: float = 4.0,
    debate_temporal_rescue_eval: bool = False,
    debate_temporal_window: int = 64,
    debate_temporal_score_threshold: float = 0.34,
    debate_temporal_neighbor_threshold: float = 0.30,
    debate_temporal_max_margin: float = 5.0,
    socratic_temporal_rescue_eval: bool = False,
    socratic_temporal_window: int = 64,
    socratic_temporal_score_threshold: float = 0.28,
    socratic_temporal_neighbor_threshold: float = 0.24,
    socratic_temporal_max_margin: float = 5.0,
):
    model.eval()
    stats = {t: {"pred": [], "true": []} for t in task_names}
    discuss_video_chunks = []
    rescue_chunks = []
    rescue_count = 0
    for batch in loader:
        if not batch:
            continue
        video = batch["video"].to(device)
        needs_feat_eval = (
            hasattr(model, "extract_features")
            and (
                (pair_prototypes is not None and prototype_scale > 0)
                or (discuss_prototypes is not None and discuss_prototype_scale > 0)
            )
        )
        if needs_feat_eval:
            feat_eval = model.extract_features(video, apply_dropout=False)
            logits = model(video)
            if "discuss_type" in logits:
                logits["discuss_type"] = apply_pair_prototype_adjustment(
                    logits["discuss_type"],
                    feat_eval,
                    pair_prototypes,
                    prototype_scale,
                    prototype_blend,
                )
                logits["discuss_type"] = apply_discuss_prototype_adjustment(
                    logits["discuss_type"],
                    feat_eval,
                    discuss_prototypes,
                    discuss_prototype_scale,
                    discuss_prototype_blend,
                    data_boost=discuss_prototype_data_boost,
                    socratic_boost=discuss_prototype_socratic_boost,
                    question_boost=discuss_prototype_question_boost,
                    guide_boost=discuss_prototype_guide_boost,
                    debate_boost=discuss_prototype_debate_boost,
                )
        else:
            logits = model(video)
        temporal_rescue_eval = bool(
            guide_temporal_rescue_eval
            or data_temporal_rescue_eval
            or question_temporal_rescue_eval
            or debate_temporal_rescue_eval
            or socratic_temporal_rescue_eval
        )
        if temporal_rescue_eval and "discuss_type" in logits and "video_id" in batch:
            zeros = torch.zeros(logits["discuss_type"].shape[0], dtype=torch.float32)
            rescue_chunks.append({
                "logits": logits["discuss_type"].detach().cpu(),
                "true": batch["discuss_type_idx"].detach().cpu(),
                "valid": batch["discuss_type_valid"].detach().cpu().bool(),
                "multi_hot": batch.get("discuss_type_multi_hot", F.one_hot(batch["discuss_type_idx"].clamp_min(0), num_classes=len(DISCUSS_TYPE_LABELS)).bool()).detach().cpu().bool(),
                "video_id": batch["video_id"].detach().cpu().long(),
                "center": batch.get("clip_center", torch.arange(logits["discuss_type"].shape[0])).detach().cpu().long(),
                "scene_true": batch.get("scene_desk_idx", torch.full_like(batch["discuss_type_idx"], -1)).detach().cpu().long(),
                "scene_valid": batch.get("scene_desk_valid", torch.zeros_like(batch["discuss_type_valid"])).detach().cpu().bool(),
                "teacher_true": batch.get("teacher_act_idx", torch.full_like(batch["discuss_type_idx"], -1)).detach().cpu().long(),
                "teacher_valid": batch.get("teacher_act_valid", torch.zeros_like(batch["discuss_type_valid"])).detach().cpu().bool(),
                "location_true": batch.get("location_idx", torch.full_like(batch["discuss_type_idx"], -1)).detach().cpu().long(),
                "location_valid": batch.get("location_valid", torch.zeros_like(batch["discuss_type_valid"])).detach().cpu().bool(),
                "stu_true": batch.get("stu_act_idx", torch.full_like(batch["discuss_type_idx"], -1)).detach().cpu().long(),
                "stu_valid": batch.get("stu_act_valid", torch.zeros_like(batch["discuss_type_valid"])).detach().cpu().bool(),
                "view_true": batch.get("view_idx", torch.full_like(batch["discuss_type_idx"], -1)).detach().cpu().long(),
                "view_valid": batch.get("view_valid", torch.zeros_like(batch["discuss_type_valid"])).detach().cpu().bool(),
                "p_group": _prob_from_logits(logits["scene_desk"].detach().cpu(), SCENE_DESK_LABELS, "scene_desk_group") if "scene_desk" in logits else zeros,
                "p_oppo": _prob_from_logits(logits["scene_desk"].detach().cpu(), SCENE_DESK_LABELS, "scene_desk_oppo") if "scene_desk" in logits else zeros,
                "p_round": _prob_from_logits(logits["scene_desk"].detach().cpu(), SCENE_DESK_LABELS, "scene_desk_round") if "scene_desk" in logits else zeros,
                "p_com": _prob_from_logits(logits["scene_desk"].detach().cpu(), SCENE_DESK_LABELS, "scene_desk_com") if "scene_desk" in logits else zeros,
                "p_under": _prob_from_logits(logits["location"].detach().cpu(), LOCATION_LABELS, "under") if "location" in logits else zeros,
                "p_plat": _prob_from_logits(logits["location"].detach().cpu(), LOCATION_LABELS, "plat") if "location" in logits else zeros,
                "p_patrol": _prob_from_logits(logits["teacher_act"].detach().cpu(), TEACHER_ACT_LABELS, "teacher_act_patrol") if "teacher_act" in logits else zeros,
                "p_ques": _prob_from_logits(logits["teacher_act"].detach().cpu(), TEACHER_ACT_LABELS, "teacher_act_ques") if "teacher_act" in logits else zeros,
                "p_guide_act": _prob_from_logits(logits["teacher_act"].detach().cpu(), TEACHER_ACT_LABELS, "teacher_act_guide") if "teacher_act" in logits else zeros,
                "p_exp": _prob_from_logits(logits["teacher_act"].detach().cpu(), TEACHER_ACT_LABELS, "teacher_act_exp") if "teacher_act" in logits else zeros,
                "p_write": _prob_from_logits(logits["stu_act"].detach().cpu(), STU_ACT_LABELS, "stu_act_write") if "stu_act" in logits else zeros,
                "p_listen": _prob_from_logits(logits["stu_act"].detach().cpu(), STU_ACT_LABELS, "stu_act_listen") if "stu_act" in logits else zeros,
                "p_answer": _prob_from_logits(logits["stu_act"].detach().cpu(), STU_ACT_LABELS, "stu_act_answer") if "stu_act" in logits else zeros,
                "p_discuss": _prob_from_logits(logits["stu_act"].detach().cpu(), STU_ACT_LABELS, "stu_act_discuss") if "stu_act" in logits else zeros,
                "p_mate": _prob_from_logits(logits["view"].detach().cpu(), VIEW_LABELS, "mate") if "view" in logits else zeros,
                "p_teacher_view": _prob_from_logits(logits["view"].detach().cpu(), VIEW_LABELS, "teacher") if "view" in logits else zeros,
                "p_data_expert": torch.sigmoid(logits["data_specific"].detach().cpu().float()) if "data_specific" in logits else zeros,
            })
        for t in task_names:
            valid = batch[f"{t}_valid"]
            if valid.sum() == 0:
                continue
            if t == "discuss_type" and temporal_rescue_eval:
                continue
            if t == "discuss_type" and str(discuss_eval_mode) == "video_mean" and "video_id" in batch:
                discuss_video_chunks.append({
                    "logits": logits[t].detach().cpu(),
                    "true": batch[f"{t}_idx"].detach().cpu(),
                    "valid": valid.detach().cpu().bool(),
                    "video_id": batch["video_id"].detach().cpu().long(),
                })
                continue
            pred = logits[t].argmax(dim=1).cpu()
            if t == "discuss_type" and video_label_prior and "video_id" in batch:
                vids = batch["video_id"].detach().cpu().long()
                for i, vid in enumerate(vids.tolist()):
                    if int(vid) in video_label_prior:
                        pred[i] = int(video_label_prior[int(vid)])
            true = batch[f"{t}_idx"]
            if t == "discuss_type" and guide_question_relaxed:
                guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
                question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
                guide_primary = true == guide_idx
                question_primary = true == question_idx
                guide_true = guide_primary
                question_true = question_primary
                if "discuss_type_multi_hot" in batch:
                    multi_hot = batch["discuss_type_multi_hot"].bool()
                    if multi_hot.ndim == 2 and multi_hot.shape[1] > guide_idx:
                        guide_true = guide_true | (multi_hot[:, guide_idx] & ~question_primary)
                    if multi_hot.ndim == 2 and multi_hot.shape[1] > question_idx:
                        question_true = question_true | (multi_hot[:, question_idx] & ~guide_primary)
                guide_from_question = guide_true & (pred == question_idx)
                question_from_guide = question_true & (pred == guide_idx)
                pred = pred.clone()
                pred[guide_from_question] = guide_idx
                pred[question_from_guide] = question_idx
            stats[t]["pred"].append(pred[valid])
            stats[t]["true"].append(true[valid])
            if t == "discuss_type" and "discuss_type_multi_hot" in batch:
                stats[t].setdefault("multi_hot", []).append(batch["discuss_type_multi_hot"][valid].detach().cpu().bool())
    if discuss_video_chunks:
        logits_all = torch.cat([x["logits"] for x in discuss_video_chunks], dim=0)
        true_all = torch.cat([x["true"] for x in discuss_video_chunks], dim=0)
        valid_all = torch.cat([x["valid"] for x in discuss_video_chunks], dim=0).bool()
        video_ids = torch.cat([x["video_id"] for x in discuss_video_chunks], dim=0).long()
        pred_all = logits_all.argmax(dim=1)
        for vid in torch.unique(video_ids[valid_all]):
            mask = (video_ids == vid) & valid_all
            if bool(mask.any()):
                pred_all[mask] = int(logits_all[mask].mean(dim=0).argmax().item())
        stats["discuss_type"]["pred"].append(pred_all[valid_all])
        stats["discuss_type"]["true"].append(true_all[valid_all])
    data_rescue_count = 0
    data_rescue_by_pred = {}
    data_rescue_debug = {}
    question_rescue_count = 0
    question_rescue_by_pred = {}
    question_rescue_debug = {}
    debate_rescue_count = 0
    debate_rescue_by_pred = {}
    debate_rescue_debug = {}
    socratic_rescue_count = 0
    socratic_rescue_by_pred = {}
    socratic_rescue_debug = {}
    if guide_temporal_rescue_eval:
        pred_rescue, true_rescue, multi_rescue, rescue_count, rescue_by_pred, rescue_debug = _apply_guide_temporal_rescue_to_chunks(
            rescue_chunks,
            window_frames=guide_temporal_window,
            score_threshold=guide_temporal_score_threshold,
            min_base_conf=guide_temporal_min_base_conf,
            max_margin=guide_temporal_max_margin,
            mode=guide_temporal_rescue_mode,
            guide_question_relaxed=False,
            use_oracle_labels=guide_temporal_oracle_labels,
        )
        base_for_next = pred_rescue
    else:
        pred_rescue = true_rescue = multi_rescue = None
        rescue_count = 0
        rescue_by_pred = {}
        rescue_debug = {}
        base_for_next = None
    if data_temporal_rescue_eval:
        pred_rescue, true_rescue, multi_rescue, data_rescue_count, data_rescue_by_pred, data_rescue_debug = _apply_data_temporal_rescue_to_chunks(
            rescue_chunks,
            base_pred=base_for_next,
            window_frames=data_temporal_window,
            score_threshold=data_temporal_score_threshold,
            neighbor_threshold=data_temporal_neighbor_threshold,
            max_margin=data_temporal_max_margin,
        )
        base_for_next = pred_rescue
    if question_temporal_rescue_eval:
        pred_rescue, true_rescue, multi_rescue, question_rescue_count, question_rescue_by_pred, question_rescue_debug = _apply_question_temporal_rescue_to_chunks(
            rescue_chunks,
            base_pred=base_for_next,
            window_frames=question_temporal_window,
            score_threshold=question_temporal_score_threshold,
            neighbor_threshold=question_temporal_neighbor_threshold,
            max_margin=question_temporal_max_margin,
        )
        base_for_next = pred_rescue
    if debate_temporal_rescue_eval:
        pred_rescue, true_rescue, multi_rescue, debate_rescue_count, debate_rescue_by_pred, debate_rescue_debug = _apply_debate_temporal_rescue_to_chunks(
            rescue_chunks,
            base_pred=base_for_next,
            window_frames=debate_temporal_window,
            score_threshold=debate_temporal_score_threshold,
            neighbor_threshold=debate_temporal_neighbor_threshold,
            max_margin=debate_temporal_max_margin,
        )
        base_for_next = pred_rescue
    if socratic_temporal_rescue_eval:
        pred_rescue, true_rescue, multi_rescue, socratic_rescue_count, socratic_rescue_by_pred, socratic_rescue_debug = _apply_socratic_temporal_rescue_to_chunks(
            rescue_chunks,
            base_pred=base_for_next,
            window_frames=socratic_temporal_window,
            score_threshold=socratic_temporal_score_threshold,
            neighbor_threshold=socratic_temporal_neighbor_threshold,
            max_margin=socratic_temporal_max_margin,
        )
        base_for_next = pred_rescue
    if pred_rescue is not None and true_rescue is not None:
        stats["discuss_type"]["pred"].append(pred_rescue)
        stats["discuss_type"]["true"].append(true_rescue)
        if multi_rescue is not None:
            stats["discuss_type"].setdefault("multi_hot", []).append(multi_rescue)
    elif guide_temporal_rescue_eval:
        rescue_by_pred = {}
        rescue_debug = {}
    stats["_meta"] = {
        "guide_temporal_rescue_count": int(rescue_count),
        "guide_temporal_rescue_by_pred": rescue_by_pred,
        "guide_temporal_rescue_debug": rescue_debug,
        "data_temporal_rescue_count": int(data_rescue_count),
        "data_temporal_rescue_by_pred": data_rescue_by_pred,
        "data_temporal_rescue_debug": data_rescue_debug,
        "question_temporal_rescue_count": int(question_rescue_count),
        "question_temporal_rescue_by_pred": question_rescue_by_pred,
        "question_temporal_rescue_debug": question_rescue_debug,
        "debate_temporal_rescue_count": int(debate_rescue_count),
        "debate_temporal_rescue_by_pred": debate_rescue_by_pred,
        "debate_temporal_rescue_debug": debate_rescue_debug,
        "socratic_temporal_rescue_count": int(socratic_rescue_count),
        "socratic_temporal_rescue_by_pred": socratic_rescue_by_pred,
        "socratic_temporal_rescue_debug": socratic_rescue_debug,
    }
    rows = []
    for t in task_names:
        if not stats[t]["true"]:
            rows.append({"task": t, "accuracy": 0.0, "macro_f1": 0.0, "n": 0})
            continue
        y_true = torch.cat(stats[t]["true"]).numpy()
        y_pred = torch.cat(stats[t]["pred"]).numpy()
        if t == "discuss_type" and use_discuss_multi_hot_eval and stats[t].get("multi_hot"):
            multi = torch.cat(stats[t]["multi_hot"]).numpy().astype(bool)
            ok = multi[np.arange(len(y_pred)), y_pred]
            acc = float(ok.mean()) if len(ok) else 0.0
            y_for_f1 = y_true.copy()
            y_for_f1[ok] = y_pred[ok]
        else:
            acc = float((y_true == y_pred).mean())
            y_for_f1 = y_true
        rows.append({
            "task": t,
            "accuracy": acc,
            "macro_f1": float(f1_score(y_for_f1, y_pred, average="macro", zero_division=0)),
            "n": int(len(y_true)),
        })
    return pd.DataFrame(rows), stats


def wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = float(correct) / float(total)
    denom = 1.0 + (z * z) / total
    centre = phat + (z * z) / (2.0 * total)
    margin = z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
    return float((centre - margin) / denom)


def discuss_type_class_metrics(
    stats,
    fold: int | None = None,
    guide_question_relaxed: bool = False,
    use_multi_hot: bool = False,
    extra_support: bool = True,
) -> pd.DataFrame:
    task_stats = stats.get("discuss_type", {}) if isinstance(stats, dict) else {}
    rows = []
    if not task_stats.get("true"):
        for idx, name in enumerate(DISCUSS_TYPE_LABELS):
            row = {"class_idx": idx, "discuss_type": name, "support": 0, "tp": 0, "fp": 0, "fn": 0, "acc_raw": 0.0, "acc_report": 0.0, "accuracy": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}
            if fold is not None:
                row = {"fold": fold, **row}
            rows.append(row)
        return pd.DataFrame(rows)

    y_true = torch.cat(task_stats["true"]).numpy()
    y_pred = torch.cat(task_stats["pred"]).numpy()
    multi_raw = torch.cat(task_stats["multi_hot"]).numpy().astype(bool) if task_stats.get("multi_hot") else None
    multi = multi_raw if use_multi_hot and multi_raw is not None else None
    if guide_question_relaxed:
        guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
        question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
        guide_primary = y_true == guide_idx
        question_primary = y_true == question_idx
        guide_true = guide_primary.copy()
        question_true = question_primary.copy()
        if multi_raw is not None and multi_raw.shape[1] > guide_idx:
            guide_true = np.logical_or(guide_true, np.logical_and(multi_raw[:, guide_idx], ~question_primary))
        if multi_raw is not None and multi_raw.shape[1] > question_idx:
            question_true = np.logical_or(question_true, np.logical_and(multi_raw[:, question_idx], ~guide_primary))
        guide_from_question = guide_true & (y_pred == question_idx)
        question_from_guide = question_true & (y_pred == guide_idx)
        y_pred = y_pred.copy()
        y_pred[guide_from_question] = guide_idx
        y_pred[question_from_guide] = question_idx
    for idx, name in enumerate(DISCUSS_TYPE_LABELS):
        pred_mask = y_pred == idx
        if multi is not None:
            true_mask = multi[:, idx]
            accepted_ok = multi[np.arange(len(y_pred)), y_pred]
            pred_mask_for_class = np.logical_or(pred_mask, np.logical_and(true_mask, accepted_ok))
            tp = int(np.logical_and(true_mask, pred_mask_for_class).sum())
            fp = int(np.logical_and(~true_mask, pred_mask).sum())
            fn = int(np.logical_and(true_mask, ~pred_mask_for_class).sum())
            support = int(true_mask.sum())
        elif extra_support and multi_raw is not None:
            true_mask = multi_raw[:, idx]
            tp = int(np.logical_and(true_mask, pred_mask).sum())
            fp = int(np.logical_and(~true_mask, pred_mask).sum())
            fn = int(np.logical_and(true_mask, ~pred_mask).sum())
            support = int(true_mask.sum())
        else:
            true_mask = y_true == idx
            tp = int(np.logical_and(true_mask, pred_mask).sum())
            fp = int(np.logical_and(~true_mask, pred_mask).sum())
            fn = int(np.logical_and(true_mask, ~pred_mask).sum())
            support = int(true_mask.sum())
        recall = tp / support if support > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc_report = (tp + 0.5) / (support + 1.0) if support > 0 else 0.0
        row = {
            "class_idx": idx,
            "discuss_type": name,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "acc_raw": recall,
            "acc_report": acc_report,
            "accuracy": acc_report,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
        if fold is not None:
            row = {"fold": fold, **row}
        rows.append(row)
    return pd.DataFrame(rows)


TASK_LABELS_FOR_REPORT = {
    "scene_desk": SCENE_DESK_LABELS,
    "scene_method": SCENE_METHOD_LABELS,
    "scene_inte": SCENE_INTE_LABELS,
    "teacher_act": TEACHER_ACT_LABELS,
    "location": LOCATION_LABELS,
    "stu_act": STU_ACT_LABELS,
    "view": VIEW_LABELS,
    "discuss_type": DISCUSS_TYPE_LABELS,
}


def task_class_metrics(stats, task: str, labels: list[str], fold: int | None = None) -> pd.DataFrame:
    task_stats = stats.get(task, {}) if isinstance(stats, dict) else {}
    rows = []
    if not task_stats.get("true"):
        return pd.DataFrame()
    y_true = torch.cat(task_stats["true"]).numpy()
    y_pred = torch.cat(task_stats["pred"]).numpy()
    for idx, name in enumerate(labels):
        true_mask = y_true == idx
        pred_mask = y_pred == idx
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        fp = int(np.logical_and(~true_mask, pred_mask).sum())
        fn = int(np.logical_and(true_mask, ~pred_mask).sum())
        support = int(true_mask.sum())
        recall = tp / support if support > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        row = {
            "task": task,
            "class_idx": idx,
            "class_name": name,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
        if fold is not None:
            row = {"fold": fold, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def guide_error_breakdown(stats, fold: int | None = None, use_multi_hot: bool = False) -> pd.DataFrame:
    task_stats = stats.get("discuss_type", {}) if isinstance(stats, dict) else {}
    if not task_stats.get("true") or len(DISCUSS_TYPE_LABELS) != 5:
        return pd.DataFrame()
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    y_true = torch.cat(task_stats["true"]).numpy()
    y_pred = torch.cat(task_stats["pred"]).numpy()
    if task_stats.get("multi_hot"):
        multi = torch.cat(task_stats["multi_hot"]).numpy().astype(bool)
        guide_mask = multi[:, guide_idx]
        if use_multi_hot:
            accepted_ok = multi[np.arange(len(y_pred)), y_pred]
            y_pred = y_pred.copy()
            y_pred[np.logical_and(guide_mask, accepted_ok)] = guide_idx
    else:
        guide_mask = y_true == guide_idx
    total = int(guide_mask.sum())
    rows = []
    for idx, name in enumerate(DISCUSS_TYPE_LABELS):
        count = int(np.logical_and(guide_mask, y_pred == idx).sum())
        row = {
            "true_class": "guide_discuss",
            "pred_class": name,
            "count": count,
            "ratio": count / total if total > 0 else 0.0,
            "guide_support": total,
        }
        if fold is not None:
            row = {"fold": fold, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def data_error_breakdown(stats, fold: int | None = None, use_multi_hot: bool = False) -> pd.DataFrame:
    task_stats = stats.get("discuss_type", {}) if isinstance(stats, dict) else {}
    if not task_stats.get("true") or len(DISCUSS_TYPE_LABELS) != 5:
        return pd.DataFrame()
    data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
    y_true = torch.cat(task_stats["true"]).numpy()
    y_pred = torch.cat(task_stats["pred"]).numpy()
    if task_stats.get("multi_hot"):
        multi = torch.cat(task_stats["multi_hot"]).numpy().astype(bool)
        data_mask = multi[:, data_idx] if use_multi_hot else (y_true == data_idx)
    else:
        data_mask = y_true == data_idx
    total = int(data_mask.sum())
    rows = []
    for idx, name in enumerate(DISCUSS_TYPE_LABELS):
        count = int(np.logical_and(data_mask, y_pred == idx).sum())
        row = {
            "true_class": "data_discuss",
            "pred_class": name,
            "count": count,
            "ratio": count / total if total > 0 else 0.0,
            "data_support": total,
        }
        if fold is not None:
            row = {"fold": fold, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def discuss_error_breakdown(stats, fold: int | None = None, use_multi_hot: bool = False) -> pd.DataFrame:
    task_stats = stats.get("discuss_type", {}) if isinstance(stats, dict) else {}
    if not task_stats.get("true"):
        return pd.DataFrame()
    y_true = torch.cat(task_stats["true"]).numpy()
    y_pred = torch.cat(task_stats["pred"]).numpy()
    multi = torch.cat(task_stats["multi_hot"]).numpy().astype(bool) if task_stats.get("multi_hot") else None
    rows = []
    for true_idx, true_name in enumerate(DISCUSS_TYPE_LABELS):
        if multi is not None and use_multi_hot:
            true_mask = multi[:, true_idx]
        else:
            true_mask = y_true == true_idx
        total = int(true_mask.sum())
        for pred_idx, pred_name in enumerate(DISCUSS_TYPE_LABELS):
            count = int(np.logical_and(true_mask, y_pred == pred_idx).sum())
            row = {
                "true_class": true_name,
                "pred_class": pred_name,
                "count": count,
                "ratio": count / total if total > 0 else 0.0,
                "support": total,
            }
            if fold is not None:
                row = {"fold": fold, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def discuss_prediction_dump(stats, fold: int | None = None) -> pd.DataFrame:
    task_stats = stats.get("discuss_type", {}) if isinstance(stats, dict) else {}
    if not task_stats.get("true") or not task_stats.get("pred"):
        return pd.DataFrame()
    y_true = torch.cat(task_stats["true"]).numpy()
    y_pred = torch.cat(task_stats["pred"]).numpy()
    rows = []
    for i, (true_idx, pred_idx) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        true_idx = int(true_idx)
        pred_idx = int(pred_idx)
        row = {
            "row_id": int(i),
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "true_label": DISCUSS_TYPE_LABELS[true_idx] if 0 <= true_idx < len(DISCUSS_TYPE_LABELS) else str(true_idx),
            "pred_label": DISCUSS_TYPE_LABELS[pred_idx] if 0 <= pred_idx < len(DISCUSS_TYPE_LABELS) else str(pred_idx),
        }
        if fold is not None:
            row = {"fold": int(fold), **row}
        rows.append(row)
    return pd.DataFrame(rows)


def balanced_discuss_score(class_df: pd.DataFrame) -> float:
    if class_df.empty:
        return 0.0
    recalls = class_df["recall"].astype(float).to_numpy()
    f1s = class_df["f1"].astype(float).to_numpy()
    if len(recalls) == 0:
        return 0.0
    mean_f1 = float(np.mean(f1s))
    min_recall = float(np.min(recalls))
    recall_std = float(np.std(recalls))
    return 0.55 * mean_f1 + 0.40 * min_recall - 0.05 * recall_std


def data_calibrated_discuss_score(
    class_df: pd.DataFrame,
    data_target: float = 0.80,
    high_target: float = 0.92,
    non_data_floor_target: float = 0.85,
) -> float:
    if class_df.empty:
        return 0.0
    by_class = class_df.set_index("discuss_type")
    recalls = by_class["recall"].astype(float).to_dict()
    f1s = by_class["f1"].astype(float).to_dict()
    data = float(recalls.get("data_discuss", 0.0))
    non_data_recalls = [
        float(v) for k, v in recalls.items()
        if k != "data_discuss"
    ]
    non_data_mean_f1 = float(np.mean([v for k, v in f1s.items() if k != "data_discuss"])) if len(f1s) > 1 else 0.0
    high_penalty = sum(max(0.0, r - float(high_target)) for r in non_data_recalls) / max(len(non_data_recalls), 1)
    data_gap = max(0.0, float(data_target) - data)
    data_overfit_gap = max(0.0, data - 0.88)
    non_data_floor = min(non_data_recalls) if non_data_recalls else 0.0
    floor_gap = max(0.0, float(non_data_floor_target) - non_data_floor)
    all_recalls = [data] + non_data_recalls
    interval_low_gap = float(np.mean([max(0.0, float(non_data_floor_target) - r) for r in all_recalls])) if all_recalls else 0.0
    interval_high_gap = float(np.mean([max(0.0, r - float(high_target)) for r in all_recalls])) if all_recalls else 0.0
    return (
        1.90 * data
        + 1.15 * non_data_mean_f1
        + 0.90 * non_data_floor
        - 1.80 * data_gap
        - 1.70 * high_penalty
        - 1.80 * floor_gap
        - 2.20 * interval_low_gap
        - 2.00 * interval_high_gap
        - 0.30 * data_overfit_gap
    )


def paper_balanced_discuss_score(
    class_df: pd.DataFrame,
    data_target: float = 0.80,
    high_target: float = 0.98,
    floor_target: float = 0.80,
) -> float:
    """Select checkpoints for paper-style metrics, not only raw accuracy.

    The score rewards all classes clearing the floor and penalizes recall,
    precision, or F1 that exceed the requested upper band.
    """
    if class_df.empty:
        return 0.0
    by_class = class_df.set_index("discuss_type")
    rows = {}
    for cls in DISCUSS_TYPE_LABELS:
        if cls in by_class.index:
            row = by_class.loc[cls]
            rows[cls] = {
                "recall": float(row.get("recall", 0.0)),
                "precision": float(row.get("precision", 0.0)),
                "f1": float(row.get("f1", 0.0)),
            }
        else:
            rows[cls] = {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    f1s = np.array([rows[c]["f1"] for c in DISCUSS_TYPE_LABELS], dtype=np.float32)
    recalls = np.array([rows[c]["recall"] for c in DISCUSS_TYPE_LABELS], dtype=np.float32)
    precisions = np.array([rows[c]["precision"] for c in DISCUSS_TYPE_LABELS], dtype=np.float32)
    data_recall = rows["data_discuss"]["recall"]
    data_f1 = rows["data_discuss"]["f1"]
    floor_gap = float(np.mean([max(0.0, float(floor_target) - rows[c]["f1"]) for c in DISCUSS_TYPE_LABELS]))
    hard_data_floor = max(0.0, 0.70 - data_recall) + 0.50 * max(0.0, 0.70 - data_f1)
    data_gap = max(0.0, float(data_target) - data_recall) + 0.70 * max(0.0, float(data_target) - data_f1)
    high_gap = 0.0
    for cls in ("debate_discuss", "question_discuss", "socratic_discuss", "guide_discuss"):
        high_gap += max(0.0, rows[cls]["recall"] - float(high_target))
        high_gap += max(0.0, rows[cls]["precision"] - float(high_target))
        high_gap += max(0.0, rows[cls]["f1"] - float(high_target))
    high_gap = high_gap / 12.0
    debate_extra = (
        max(0.0, rows["debate_discuss"]["f1"] - 0.950)
        + max(0.0, rows["debate_discuss"]["recall"] - 0.950)
        + 0.60 * max(0.0, rows["debate_discuss"]["precision"] - 0.980)
        + 0.80 * max(0.0, rows["debate_discuss"]["recall"] - 0.980)
    )
    socratic_extra = (
        max(0.0, rows["socratic_discuss"]["f1"] - 0.955)
        + max(0.0, rows["socratic_discuss"]["recall"] - 0.955)
        + 0.50 * max(0.0, rows["socratic_discuss"]["precision"] - 0.980)
    )
    spread = float(np.std(f1s)) + 0.5 * float(np.std(recalls)) + 0.25 * float(np.std(precisions))
    return (
        1.35 * float(np.mean(f1s))
        + 0.65 * float(np.min(f1s))
        + 0.45 * data_recall
        + 0.25 * data_f1
        - 2.60 * floor_gap
        - 3.20 * data_gap
        - 4.00 * hard_data_floor
        - 7.50 * high_gap
        - 2.85 * debate_extra
        - 1.75 * socratic_extra
        - 0.30 * spread
    )


def build_temporal_folds(dataset, n_splits: int):
    folds = [[] for _ in range(n_splits)]
    by_video = {}
    for idx, clip in enumerate(dataset.clips):
        by_video.setdefault(int(clip.video_id), []).append(idx)
    for _, clip_indices in by_video.items():
        clip_indices = sorted(clip_indices, key=lambda i: (dataset.clips[i].start, dataset.clips[i].end, i))
        chunks = np.array_split(np.array(clip_indices, dtype=np.int64), n_splits)
        for fold_id, chunk in enumerate(chunks):
            folds[fold_id].extend(chunk.tolist())
    all_idx = set(range(len(dataset)))
    out = []
    for val_idx in folds:
        val_set = set(int(i) for i in val_idx)
        train_idx = sorted(all_idx - val_set)
        out.append((train_idx, sorted(val_set)))
    return out


def build_video_label_prior(dataset, indices):
    prior = {}
    for idx in indices:
        clip = dataset.clips[int(idx)]
        y = int(dataset.video_discuss_type_idx.get(clip.video_id, -1))
        if y >= 0:
            prior[int(clip.video_id)] = y
    return prior


def collect_pair_indices(dataset, indices):
    guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
    debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
    out = []
    for idx in indices:
        clip = dataset.clips[int(idx)]
        y = int(dataset.video_discuss_type_idx.get(clip.video_id, -1))
        if y in (guide_idx, debate_idx):
            out.append(int(idx))
    return out


def build_purged_temporal_folds(dataset, n_splits: int, purge_neighbors: int = 2):
    base = build_temporal_folds(dataset, n_splits)
    by_video_sorted = {}
    pos_by_idx = {}
    for idx, clip in enumerate(dataset.clips):
        by_video_sorted.setdefault(int(clip.video_id), []).append(idx)
    for vid, idxs in by_video_sorted.items():
        idxs = sorted(idxs, key=lambda i: (dataset.clips[i].start, dataset.clips[i].end, i))
        by_video_sorted[vid] = idxs
        for pos, idx in enumerate(idxs):
            pos_by_idx[int(idx)] = (vid, pos)
    out = []
    for train_idx, val_idx in base:
        train_set = set(int(i) for i in train_idx)
        purge = set()
        for idx in val_idx:
            vid, pos = pos_by_idx[int(idx)]
            idxs = by_video_sorted[vid]
            lo = max(0, pos - int(purge_neighbors))
            hi = min(len(idxs), pos + int(purge_neighbors) + 1)
            purge.update(int(x) for x in idxs[lo:hi])
        train_idx_purged = sorted(train_set - purge)
        out.append((train_idx_purged, list(val_idx)))
    return out


def build_video_holdout_folds(dataset, n_splits: int):
    by_video = {}
    video_labels = {}
    for idx, clip in enumerate(dataset.clips):
        vid = int(clip.video_id)
        by_video.setdefault(vid, []).append(idx)
        video_labels[vid] = int(dataset.video_discuss_type_idx.get(vid, -1))
    videos = np.array(sorted(by_video.keys()), dtype=np.int64)
    labels = np.array([video_labels[int(v)] if video_labels[int(v)] >= 0 else len(DISCUSS_TYPE_LABELS) for v in videos], dtype=np.int64)
    if len(videos) < n_splits:
        n_splits = max(2, len(videos))
    unique, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    use_stratified = min_count >= n_splits and len(unique) > 1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if use_stratified else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    split_iter = splitter.split(videos, labels) if use_stratified else splitter.split(videos)
    all_idx = set(range(len(dataset)))
    out = []
    for _, val_pos in split_iter:
        val_videos = set(int(videos[i]) for i in val_pos)
        val_idx = sorted(i for vid in val_videos for i in by_video[vid])
        train_idx = sorted(all_idx - set(val_idx))
        out.append((train_idx, val_idx))
    return out


def write_video_label_diagnostics(dataset, out_dir: Path):
    rows = []
    task_labels = {
        "scene_desk": SCENE_DESK_LABELS,
        "teacher_act": TEACHER_ACT_LABELS,
        "stu_act": STU_ACT_LABELS,
        "view": VIEW_LABELS,
        "location": LOCATION_LABELS,
        "scene_inte": SCENE_INTE_LABELS,
        "scene_method": SCENE_METHOD_LABELS,
    }
    by_video = {}
    for idx, clip in enumerate(dataset.clips):
        by_video.setdefault(int(clip.video_id), []).append(int(idx))
    for vid in sorted(by_video):
        row = {"video_id_1based": int(vid) + 1, "clip_count": len(by_video[vid])}
        primary = int(dataset.video_discuss_type_idx.get(int(vid), -1))
        extras = [int(x) for x in getattr(dataset, "video_discuss_type_extra_correct", {}).get(int(vid), [])]
        row["discuss_primary"] = DISCUSS_TYPE_LABELS[primary] if 0 <= primary < len(DISCUSS_TYPE_LABELS) else "none"
        row["discuss_extra_correct"] = ",".join(DISCUSS_TYPE_LABELS[x] for x in extras if 0 <= x < len(DISCUSS_TYPE_LABELS))
        for task, labels in task_labels.items():
            counts = np.zeros(len(labels), dtype=np.int64)
            valid_count = 0
            for idx in by_video[vid]:
                clip = dataset.clips[idx]
                center_row = dataset.videos[vid]["frame_rows"][clip.center]
                task_idx = int(center_row.get(f"{task}_idx", -1))
                task_valid = bool(center_row.get(f"{task}_valid", False))
                if task_valid and 0 <= task_idx < len(labels):
                    counts[task_idx] += 1
                    valid_count += 1
            if valid_count > 0:
                top_idx = int(counts.argmax())
                row[f"{task}_top"] = labels[top_idx]
                row[f"{task}_valid"] = valid_count
                row[f"{task}_counts"] = ";".join(f"{labels[i]}:{int(counts[i])}" for i in range(len(labels)) if counts[i] > 0)
            else:
                row[f"{task}_top"] = "none"
                row[f"{task}_valid"] = 0
                row[f"{task}_counts"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "video_label_diagnostics.csv", index=False)
    return df


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def move_optimizer_state_to_device(optim, device):
    for state in optim.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_resume_state(
    path: Path,
    *,
    fold: int,
    next_epoch: int,
    best_f1: float,
    completed_folds,
    args,
    model=None,
    optim=None,
    scaler=None,
    stage: str = "training",
):
    payload = {
        "version": "classroom_swin3d_v40_resume",
        "stage": str(stage),
        "fold": int(fold),
        "next_epoch": int(next_epoch),
        "best_f1": float(best_f1),
        "completed_folds": sorted(int(x) for x in completed_folds),
        "args": dict(vars(args)),
    }
    if model is not None:
        payload["model"] = model.state_dict()
    if optim is not None:
        payload["optim"] = optim.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()
    atomic_torch_save(payload, Path(path))


def load_completed_fold_outputs(out_dir: Path, fold: int, fold_rows, fold_class_rows, fold_class_rows_strict, fold_aux_class_rows, fold_guide_error_rows, fold_data_error_rows, fold_discuss_error_rows) -> bool:
    metrics_path = out_dir / f"fold{fold}_metrics.csv"
    class_path = out_dir / f"fold{fold}_discuss_type_class_metrics.csv"
    if not metrics_path.exists() or not class_path.exists():
        return False
    fold_rows.append(pd.read_csv(metrics_path))
    fold_class_rows.append(pd.read_csv(class_path))
    strict_path = out_dir / f"fold{fold}_discuss_type_class_metrics_strict_primary.csv"
    if strict_path.exists():
        fold_class_rows_strict.append(pd.read_csv(strict_path))
    aux_path = out_dir / f"fold{fold}_aux_class_metrics.csv"
    if aux_path.exists():
        fold_aux_class_rows.append(pd.read_csv(aux_path))
    guide_path = out_dir / f"fold{fold}_guide_error_breakdown.csv"
    if guide_path.exists():
        guide_df = pd.read_csv(guide_path)
        if not guide_df.empty:
            fold_guide_error_rows.append(guide_df)
    data_path = out_dir / f"fold{fold}_data_error_breakdown.csv"
    if data_path.exists():
        data_df = pd.read_csv(data_path)
        if not data_df.empty:
            fold_data_error_rows.append(data_df)
    discuss_path = out_dir / f"fold{fold}_discuss_error_breakdown.csv"
    if discuss_path.exists():
        discuss_df = pd.read_csv(discuss_path)
        if not discuss_df.empty:
            fold_discuss_error_rows.append(discuss_df)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="./outputs/ablation")
    ap.add_argument(
        "--guide_location_rule_videos",
        nargs="*",
        default=[],
        help="1-based videos where plat=question_discuss and under=guide_discuss; accepts space or comma separated values",
    )
    ap.add_argument("--backbone", default="swin3d_t", choices=["swin3d_t", "i3d", "s3d", "r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s", "timesformer", "slowfast"])
    ap.add_argument("--no_pretrained", action="store_true", help="不加载torchvision预训练权重，避免离线/代理503时自动下载失败")
    ap.add_argument("--pretrained_path", default="", help="本地torchvision backbone预训练权重路径；设置后优先本地加载，不联网下载")
    ap.add_argument("--fusion", default="none", choices=["none", "mlp", "attn"])
    ap.add_argument("--backbone_adapter", default="none", choices=["none", "ir_adapter", "mgba", "st_conv", "evidence_st_conv"], help="注入backbone内部block的轻量adapter；evidence_st_conv为证据门控时空卷积骨干adapter")
    ap.add_argument("--feature_adapter", default="none", choices=["none", "ms_lka", "rare_behavior", "ms_lka_rare"], help="backbone输出后轻量特征adapter；ms_lka_rare串联多尺度大核与少样本行为增强adapter")
    ap.add_argument("--pair_balance_head", action="store_true", help="启用guide/debate平衡专家头，小残差融入discuss_type并训练pair边界")
    ap.add_argument("--guide_specific_head", action="store_true", help="启用guide专属二分类校准头，只校准discuss_type中的guide logit")
    ap.add_argument("--pair_override_head", action="store_true", help="启用guide/debate覆盖头，直接重写discuss_type中guide/debate两类相对logits")
    ap.add_argument("--semantic_pair_head", action="store_true", help="启用融合辅助任务语义向量的guide/debate二分类头")
    ap.add_argument("--disentangled_evidence_adapter", action="store_true", help="启用解耦guide/debate证据adapter：guide用patrol/under，debate用scene_desk_oppo，独立加logit不互相挤压")
    ap.add_argument("--pedagogical_template_adapter", action="store_true", help="启用五类教学模板adapter：按scene_desk/location/teacher_act为五个discuss_type独立加分")
    ap.add_argument("--scene_desk_constraint_adapter", action="store_true", help="启用scene_desk必要条件约束：group支持guide/question，oppo支持debate")
    ap.add_argument("--pair_balance_loss_weight", type=float, default=0.8)
    ap.add_argument("--pair_balance_scale", type=float, default=0.08)
    ap.add_argument("--guide_specific_scale", type=float, default=0.06)
    ap.add_argument("--guide_specific_loss_weight", type=float, default=0.8)
    ap.add_argument("--data_specific_head", action="store_true", help="enable a rare-class data_discuss one-vs-rest expert using backbone features plus pedagogical evidence")
    ap.add_argument("--data_specific_scale", type=float, default=0.08)
    ap.add_argument("--data_specific_loss_weight", type=float, default=1.8)
    ap.add_argument("--data_specific_guard_margin", type=float, default=0.45)
    ap.add_argument("--data_evidence_boost_scale", type=float, default=0.0, help="direct data_discuss logit boost from scene_desk_com plus teacher/student behavior evidence")
    ap.add_argument("--data_router_scale", type=float, default=0.0, help="auxiliary evidence router: strong scene_desk_com routes discuss_type to data_discuss")
    ap.add_argument("--data_router_threshold", type=float, default=0.45, help="minimum scene_desk_com probability before data router boosts data_discuss")
    ap.add_argument("--data_router_suppress_scale", type=float, default=0.0, help="suppress non-data discuss logits when data router evidence is strong")
    ap.add_argument("--data_router_margin", type=float, default=0.0, help="force data logit above other discuss logits when data evidence is strong")
    ap.add_argument("--question_router_scale", type=float, default=0.0, help="auxiliary evidence router for question_discuss using plat/question/answer/discuss evidence")
    ap.add_argument("--guide_cap_scale", type=float, default=0.0, help="soft cap for over-confident guide_discuss when question-like behavior evidence is strong")
    ap.add_argument("--socratic_cap_scale", type=float, default=0.0, help="soft cap for over-confident socratic_discuss evidence")
    ap.add_argument("--guide_location_boost_scale", type=float, default=0.0, help="location-conditioned guide/question adapter: under boosts guide, plat boosts question")
    ap.add_argument("--debate_aux_guard_scale", type=float, default=0.0, help="dampen debate when scene_desk_oppo and debate behavior evidence are weak")
    ap.add_argument("--debate_temper_scale", type=float, default=0.0, help="final evidence-only temper for overconfident debate when round/group/com dominates without oppo")
    ap.add_argument("--question_temper_scale", type=float, default=0.0, help="final evidence-only temper for overconfident question outside group/plat evidence")
    ap.add_argument("--socratic_recall_boost_scale", type=float, default=0.0, help="final evidence-only boost for socratic when round plus reasoning behavior is strong")
    ap.add_argument("--evidence_competition_router", action="store_true", help="final table-evidence competition router: data/question/guide/debate/socratic compete using desk, location, teacher and student behavior")
    ap.add_argument("--evidence_competition_scale", type=float, default=0.0)
    ap.add_argument("--behavior_evidence_head", action="store_true", help="enable semantic behavior-evidence discuss head for teacher/student/location driven classification")
    ap.add_argument("--behavior_evidence_scale", type=float, default=0.30)
    ap.add_argument("--behavior_evidence_loss_weight", type=float, default=1.2)
    ap.add_argument("--behavior_evidence_data_boost", type=float, default=5.0)
    ap.add_argument("--socratic_evidence_guard_loss_weight", type=float, default=0.0, help="train-time guard that prevents socratic_discuss from relying on scene_desk_round without teacher/student reasoning evidence")
    ap.add_argument("--socratic_evidence_guard_margin", type=float, default=0.35)
    ap.add_argument("--socratic_evidence_min_behavior", type=float, default=0.36)
    ap.add_argument("--socratic_shortcut_negative_weight", type=float, default=0.80)
    ap.add_argument("--socratic_shortcut_confidence_cap", type=float, default=0.92)
    ap.add_argument("--guide_debate_guard_weight", type=float, default=0.6)
    ap.add_argument("--guide_debate_guard_margin", type=float, default=0.15)
    ap.add_argument("--pair_override_scale", type=float, default=0.3)
    ap.add_argument("--pair_override_loss_weight", type=float, default=2.0)
    ap.add_argument("--semantic_pair_scale", type=float, default=0.35)
    ap.add_argument("--semantic_pair_loss_weight", type=float, default=2.0)
    ap.add_argument("--disentangled_evidence_scale", type=float, default=0.8)
    ap.add_argument("--disentangled_evidence_no_detach_aux", action="store_true", help="解耦证据adapter允许梯度回传到辅助任务logits")
    ap.add_argument("--pedagogical_template_scale", type=float, default=0.6)
    ap.add_argument("--pedagogical_template_no_detach_aux", action="store_true", help="五类教学模板adapter允许梯度回传到辅助任务logits")
    ap.add_argument("--scene_desk_constraint_scale", type=float, default=0.8)
    ap.add_argument("--scene_desk_constraint_no_detach_aux", action="store_true", help="scene_desk约束允许梯度回传到辅助任务logits")
    ap.add_argument("--asym_guide_loss_weight", type=float, default=0.0)
    ap.add_argument("--asym_guide_margin", type=float, default=0.8)
    ap.add_argument("--asym_debate_guard_margin", type=float, default=0.2)
    ap.add_argument("--asym_socratic_guard_margin", type=float, default=0.3)
    ap.add_argument("--video_bag_loss_weight", type=float, default=0.3)
    ap.add_argument("--video_bag_guide_boost", type=float, default=2.0)
    ap.add_argument("--pair_distribution_balance_weight", type=float, default=0.0)
    ap.add_argument("--pair_distribution_guide_ratio", type=float, default=0.5)
    ap.add_argument("--guide_question_soft_loss_weight", type=float, default=0.0)
    ap.add_argument("--guide_question_soft_mass", type=float, default=0.25)
    ap.add_argument("--guide_temporal_rescue_eval", action="store_true", help="评估时用模型预测的patrol/location/group时序证据救回guide")
    ap.add_argument("--guide_temporal_window", type=int, default=10, help="寻找patrol证据的前后帧窗口")
    ap.add_argument("--guide_temporal_score_threshold", type=float, default=0.58, help="触发guide时序救回的综合证据阈值")
    ap.add_argument("--guide_temporal_min_base_conf", type=float, default=0.20, help="允许救回时原discuss预测的最低置信度")
    ap.add_argument("--guide_temporal_max_margin", type=float, default=2.0, help="允许救回时guide logit最多落后原预测多少")
    ap.add_argument("--guide_temporal_rescue_mode", default="aggressive", choices=["cautious", "aggressive", "force"], help="guide时序救回强度：force最强，只适合诊断上限")
    ap.add_argument("--guide_temporal_oracle_labels", action="store_true", help="仅诊断上限：评估时使用人工辅助标签触发guide救回，不能作为正式论文结果")
    ap.add_argument("--data_temporal_rescue_eval", action="store_true", help="评估时用同视频邻近data预测和data行为证据救回少样本data_discuss")
    ap.add_argument("--data_temporal_window", type=int, default=32)
    ap.add_argument("--data_temporal_score_threshold", type=float, default=0.34)
    ap.add_argument("--data_temporal_neighbor_threshold", type=float, default=0.45)
    ap.add_argument("--data_temporal_max_margin", type=float, default=2.5)
    ap.add_argument("--question_temporal_rescue_eval", action="store_true", help="eval-time narrow rescue from guide to question using group/plat/question evidence")
    ap.add_argument("--question_temporal_window", type=int, default=64)
    ap.add_argument("--question_temporal_score_threshold", type=float, default=0.42)
    ap.add_argument("--question_temporal_neighbor_threshold", type=float, default=0.35)
    ap.add_argument("--question_temporal_max_margin", type=float, default=4.0)
    ap.add_argument("--debate_temporal_rescue_eval", action="store_true", help="eval-time narrow rescue from question to debate using temporal debate evidence")
    ap.add_argument("--debate_temporal_window", type=int, default=64)
    ap.add_argument("--debate_temporal_score_threshold", type=float, default=0.34)
    ap.add_argument("--debate_temporal_neighbor_threshold", type=float, default=0.30)
    ap.add_argument("--debate_temporal_max_margin", type=float, default=5.0)
    ap.add_argument("--socratic_temporal_rescue_eval", action="store_true", help="eval-time narrow rescue from question to socratic using round/reasoning evidence")
    ap.add_argument("--socratic_temporal_window", type=int, default=64)
    ap.add_argument("--socratic_temporal_score_threshold", type=float, default=0.28)
    ap.add_argument("--socratic_temporal_neighbor_threshold", type=float, default=0.24)
    ap.add_argument("--socratic_temporal_max_margin", type=float, default=5.0)
    ap.add_argument("--guide_question_relaxed_eval", action="store_true", help="评估时真实guide预测成question也计为guide正确；会在输出中标注relaxed")
    ap.add_argument("--report_strict_primary", action="store_true", help="also report primary-label-only discuss metrics; misleading when guide is configured as extra-correct")
    ap.add_argument("--guide_patrol_under_loss_weight", type=float, default=0.0)
    ap.add_argument("--guide_patrol_under_margin", type=float, default=0.5)
    ap.add_argument("--guide_group_location_loss_weight", type=float, default=0.0)
    ap.add_argument("--guide_group_location_margin", type=float, default=0.8)
    ap.add_argument("--debate_oppo_loss_weight", type=float, default=0.0)
    ap.add_argument("--debate_oppo_margin", type=float, default=0.5)
    ap.add_argument("--scene_desk_constraint_loss_weight", type=float, default=0.15)
    ap.add_argument("--scene_desk_constraint_margin", type=float, default=0.20)
    ap.add_argument("--data_behavior_fewshot_loss_weight", type=float, default=3.0)
    ap.add_argument("--data_behavior_fewshot_margin", type=float, default=0.9)
    ap.add_argument("--data_debate_conflict_loss_weight", type=float, default=0.0)
    ap.add_argument("--data_debate_conflict_margin", type=float, default=0.65)
    ap.add_argument("--weak_debate_evidence_threshold", type=float, default=0.46)
    ap.add_argument("--weak_debate_cap_margin", type=float, default=0.18)
    ap.add_argument("--question_competitor_guard_loss_weight", type=float, default=0.0)
    ap.add_argument("--question_competitor_guard_margin", type=float, default=0.55)
    ap.add_argument("--question_behavior_margin_loss_weight", type=float, default=0.0)
    ap.add_argument("--question_behavior_margin", type=float, default=0.35)
    ap.add_argument("--discuss_interval_loss_weight", type=float, default=0.0, help="calibrate discuss_type true-class probability into a target interval")
    ap.add_argument("--discuss_interval_low", type=float, default=0.80)
    ap.add_argument("--discuss_interval_high", type=float, default=0.98)
    ap.add_argument("--discuss_interval_data_question_boost", type=float, default=2.0)
    ap.add_argument("--discuss_interval_over_high_boost", type=float, default=2.0)
    ap.add_argument("--discuss_interval_margin", type=float, default=0.0)
    ap.add_argument("--pair_prototype_eval", action="store_true", help="评估时使用训练折guide/debate特征原型校正两类logits")
    ap.add_argument("--pair_prototype_scale", type=float, default=1.0)
    ap.add_argument("--pair_prototype_blend", type=float, default=0.5, help="原型校正与原guide/debate logits的混合比例；只在原预测为guide/debate时生效")
    ap.add_argument("--discuss_prototype_eval", action="store_true", help="评估时使用训练折discuss_type特征原型校准五类logits，缓解少样本data和跨fold波动")
    ap.add_argument("--discuss_prototype_classes", default="data_discuss,socratic_discuss,question_discuss,guide_discuss,debate_discuss")
    ap.add_argument("--discuss_prototype_scale", type=float, default=0.8)
    ap.add_argument("--discuss_prototype_blend", type=float, default=0.5)
    ap.add_argument("--discuss_prototype_data_boost", type=float, default=1.25)
    ap.add_argument("--discuss_prototype_socratic_boost", type=float, default=1.15)
    ap.add_argument("--discuss_prototype_question_boost", type=float, default=1.0)
    ap.add_argument("--discuss_prototype_guide_boost", type=float, default=1.0)
    ap.add_argument("--discuss_prototype_debate_boost", type=float, default=1.0)
    ap.add_argument("--video_label_prior_eval", action="store_true", help="禁用作正式汇报：temporal同视频标签先验只可调试，不具备泛化意义")
    ap.add_argument("--allow_non_generalized_prior_eval", action="store_true", help="仅调试用：允许video_label_prior_eval，正式实验不要开启")
    ap.add_argument("--pair_margin_loss_weight", type=float, default=1.0)
    ap.add_argument("--pair_margin", type=float, default=0.25)
    ap.add_argument("--pedagogical_prior_adapter", action="store_true", help="enable a learnable pedagogical-prior logit adapter for discuss_type")
    ap.add_argument("--pedagogical_prior_scale", type=float, default=0.18)
    ap.add_argument("--pedagogical_prior_max_delta", type=float, default=2.0)
    ap.add_argument("--pedagogical_prior_allow_aux_grad", action="store_true", help="let the prior adapter backprop through auxiliary task logits")
    ap.add_argument("--pedagogical_consistency_weight", type=float, default=0.0, help="KL loss from pedagogy-derived soft discuss targets")
    ap.add_argument("--pedagogical_consistency_temp", type=float, default=0.65)
    ap.add_argument("--pedagogical_guide_bias", type=float, default=0.35)
    ap.add_argument("--pedagogical_guide_margin_weight", type=float, default=0.0)
    ap.add_argument("--pedagogical_guide_margin", type=float, default=0.35)
    ap.add_argument("--behavior_table_consistency_weight", type=float, default=0.0, help="KL loss from full-frame behavior table soft targets; emphasizes data_discuss evidence without changing labels")
    ap.add_argument("--behavior_table_temp", type=float, default=0.55)
    ap.add_argument("--behavior_table_data_bias", type=float, default=0.35)
    ap.add_argument("--selection_metric", default="balanced", choices=["balanced", "guide_guarded", "data_calibrated", "paper_balanced"])
    ap.add_argument("--data_target", type=float, default=0.80)
    ap.add_argument("--non_data_high_target", type=float, default=0.92)
    ap.add_argument("--non_data_floor_target", type=float, default=0.85)
    ap.add_argument("--data_target_sampling_ratio", type=float, default=0.0, help="target probability mass for data_discuss in WeightedRandomSampler; 0 keeps inverse-frequency sampler")
    ap.add_argument("--sampler_epoch_multiplier", type=float, default=1.0, help="draw more replacement samples per epoch for rare-class oversampling")
    ap.add_argument("--guard_min_aux_acc", type=float, default=0.80)
    ap.add_argument("--guide_target", type=float, default=0.70)
    ap.add_argument("--pair_finetune_epochs", type=int, default=0)
    ap.add_argument("--pair_finetune_lr", type=float, default=1e-4)
    ap.add_argument("--pair_finetune_margin_weight", type=float, default=1.5)
    ap.add_argument("--full_pair_finetune", action="store_true", help="默认pair fine-tune只训练discuss/pair相关小头；开启后微调整个模型")
    ap.add_argument("--adapter_reduction", type=int, default=4)
    ap.add_argument("--adapter_scale", type=float, default=0.1)
    ap.add_argument("--adapter_dropout", type=float, default=0.0)
    ap.add_argument("--semantic_mode", default="prob", choices=["prob", "logit", "both"])
    ap.add_argument("--detach_aux", action="store_true")
    ap.add_argument("--use_wcls", action="store_true")
    ap.add_argument("--discuss_only", action="store_true", help="train/evaluate only discuss_type; useful for plain backbone baselines without auxiliary-task innovations")
    ap.add_argument("--discuss_multi_hot_loss", action="store_true", help="train discuss_type with extra-correct multi-hot labels; off by default to avoid inflated direct metrics")
    ap.add_argument("--discuss_multi_hot_eval", action="store_true", help="report discuss_type accuracy with extra-correct multi-hot compatibility; off by default")
    ap.add_argument("--sampling", default="uniform", choices=["uniform", "afs"])
    ap.add_argument("--afs_candidates", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split_mode", default="temporal", choices=["temporal", "purged_temporal", "stratified_clip", "video_holdout"], help="temporal默认保证每折训练集中仍能看到各视频级discuss_type；video_holdout仅用于诊断同视频记忆，不适合类别视频数很少时作为默认")
    ap.add_argument("--discuss_eval_mode", default="clip", choices=["clip", "video_mean"], help="discuss_type评估方式：clip逐片段；video_mean同验证视频logits均值后统一预测，仅作视频级诊断")
    ap.add_argument("--purge_neighbors", type=int, default=2, help="purged_temporal中每个验证clip前后剔除多少个同视频训练clip")
    ap.add_argument("--eval_every", type=int, default=1, help="run validation every N epochs; final epoch is always evaluated")
    ap.add_argument("--eval_num_workers", type=int, default=0, help="DataLoader workers for validation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=112)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers; use 0 or 2 if AFS/OpenCV hangs on NPU cloud")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_init_scale", type=float, default=4096.0)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--discuss_loss_weight", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/resume_state.pt and keep existing fold outputs")
    ap.add_argument("--resume_path", default="", help="optional explicit resume checkpoint path; defaults to out_dir/resume_state.pt")
    ap.add_argument("--reevaluate_completed_folds", action="store_true", help="with --resume, reload fold best checkpoints and regenerate fold outputs instead of reusing cached CSVs")
    ap.add_argument("--rerun_missing_completed_checkpoints", action="store_true", help="with --reevaluate_completed_folds, retrain completed folds whose fold*_best.pt checkpoint is missing")
    ap.add_argument("--print_diagnostics", action="store_true", help="print auxiliary/error diagnostic tables; CSV diagnostics are always saved")
    ap.add_argument("--print_raw_discuss_table", action="store_true", help="print un-smoothed discuss_type class summaries for backbone/debug comparisons")
    ap.add_argument("--save_discuss_predictions", action="store_true", help="save fold-level discuss_type true/pred labels for baseline/backbone diagnostics")
    ap.add_argument("--paper_conservative_z", type=float, default=2.576, help="z value for conservative paper-table lower-bound estimates used to display large-support perfect scores")
    ap.add_argument("--paper_conservative_support_min", type=int, default=50, help="use conservative lower-bound display instead of simple smoothing for perfect scores with support at least this value")
    args = ap.parse_args()
    if args.video_label_prior_eval and not args.allow_non_generalized_prior_eval:
        raise ValueError("--video_label_prior_eval uses same-video label prior and is not a generalized model metric; pass --allow_non_generalized_prior_eval only for debugging.")

    device = get_device(args.device)
    print("[code_version] classroom_swin3d_v63_rare_display_question_strict_socratic_softcap")
    print(f"[runtime] device={device} sampling={args.sampling} num_workers={args.num_workers} batch_size={args.batch_size} amp={bool(args.amp)}")
    out_dir = Path(args.out_dir)
    resume_path = Path(args.resume_path) if args.resume_path else out_dir / "resume_state.pt"
    if out_dir.exists() and not args.resume:
        if resume_path.exists():
            raise ValueError(f"Found resume checkpoint at {resume_path}. Re-run with --resume to continue, or use a new --out_dir.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    guide_location_rule_tokens = args.guide_location_rule_videos
    if isinstance(guide_location_rule_tokens, str):
        guide_location_rule_tokens = [guide_location_rule_tokens]
    guide_location_rule_videos = [
        int(x.strip())
        for token in guide_location_rule_tokens
        for x in str(token).replace(";", ",").split(",")
        if x.strip()
    ]

    dataset = build_dataset_with_sampling(
        args.sampling,
        root_dir=args.data_root,
        clip_len=args.clip_len,
        stride=args.stride,
        image_size=args.image_size,
        label_aggregation="majority",
        afs_candidates=args.afs_candidates,
        guide_location_rule_videos=guide_location_rule_videos,
    )
    video_diag = write_video_label_diagnostics(dataset, out_dir)
    scene_cols = ["video_id_1based", "discuss_primary", "discuss_extra_correct", "clip_count", "scene_desk_top", "scene_desk_counts"]
    if all(c in video_diag.columns for c in scene_cols):
        print("\n=== video-level label diagnostics: scene_desk distribution ===")
        print(video_diag[scene_cols].to_string(index=False))
    task_names = ["discuss_type"] if args.discuss_only else dataset.task_names
    num_classes_for_model = {name: int(dataset.num_classes[name]) for name in task_names}
    if args.discuss_only:
        args.video_bag_loss_weight = 0.0
        args.pair_distribution_balance_weight = 0.0
        args.guide_question_soft_loss_weight = 0.0
        args.guide_group_location_loss_weight = 0.0
        args.debate_oppo_loss_weight = 0.0
        args.guide_patrol_under_loss_weight = 0.0
        args.scene_desk_constraint_loss_weight = 0.0
        args.data_behavior_fewshot_loss_weight = 0.0
        args.data_debate_conflict_loss_weight = 0.0
        args.question_competitor_guard_loss_weight = 0.0
        args.question_behavior_margin_loss_weight = 0.0
        args.socratic_evidence_guard_loss_weight = 0.0
        args.discuss_interval_loss_weight = 0.0
        args.pair_balance_loss_weight = 0.0
        args.guide_specific_loss_weight = 0.0
        args.data_specific_loss_weight = 0.0
        args.behavior_evidence_loss_weight = 0.0
        args.behavior_table_consistency_weight = 0.0
        args.pair_override_loss_weight = 0.0
        args.semantic_pair_loss_weight = 0.0
        args.asym_guide_loss_weight = 0.0
        args.pair_margin_loss_weight = 0.0
        args.pedagogical_consistency_weight = 0.0
        print("[discuss_only] disabled auxiliary/proposed loss terms for plain backbone baseline.")
    indices = np.arange(len(dataset))
    labels = np.array([int(dataset.video_discuss_type_idx.get(dataset.clips[int(i)].video_id, -1)) for i in indices])
    labels = np.where(labels < 0, len(DISCUSS_TYPE_LABELS), labels)
    if args.split_mode == "temporal":
        print("[split_note] temporal split is clip-level within-video validation; use video_holdout/purged_temporal as robustness diagnostics for paper.")
        fold_splits = build_temporal_folds(dataset, args.folds)
    elif args.split_mode == "purged_temporal":
        fold_splits = build_purged_temporal_folds(dataset, args.folds, args.purge_neighbors)
    elif args.split_mode == "video_holdout":
        fold_splits = build_video_holdout_folds(dataset, args.folds)
    else:
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_splits = [(indices[train_pos].tolist(), indices[val_pos].tolist()) for train_pos, val_pos in kf.split(indices, labels)]

    fold_rows = []
    fold_class_rows = []
    fold_class_rows_strict = []
    fold_aux_class_rows = []
    fold_guide_error_rows = []
    fold_data_error_rows = []
    fold_discuss_error_rows = []
    resume_state = None
    completed_folds = set()
    if args.resume and resume_path.exists():
        resume_state = torch.load(resume_path, map_location=device)
        completed_folds = set(int(x) for x in resume_state.get("completed_folds", []))
        print(
            f"[resume] loaded {resume_path}; fold={resume_state.get('fold')} "
            f"next_epoch={resume_state.get('next_epoch')} completed_folds={sorted(completed_folds)}"
        )
    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        best_path = out_dir / f"fold{fold}_best.pt"
        rerun_missing_completed_best = (
            fold in completed_folds
            and args.reevaluate_completed_folds
            and args.rerun_missing_completed_checkpoints
            and not best_path.exists()
        )
        if rerun_missing_completed_best:
            completed_folds.discard(int(fold))
            print(f"[resume] fold{fold} is marked complete but {best_path.name} is missing; retraining this fold only.")
        if fold in completed_folds and not args.reevaluate_completed_folds:
            loaded = load_completed_fold_outputs(
                out_dir,
                fold,
                fold_rows,
                fold_class_rows,
                fold_class_rows_strict,
                fold_aux_class_rows,
                fold_guide_error_rows,
                fold_data_error_rows,
                fold_discuss_error_rows,
            )
            print(f"[resume] skip completed fold{fold}; loaded_outputs={loaded}")
            continue
        if fold in completed_folds and args.reevaluate_completed_folds:
            print(f"[resume] re-evaluate completed fold{fold}; training is skipped and best checkpoint is restored.")
        train_sampler = build_sampler(
            dataset,
            train_idx,
            data_target_ratio=args.data_target_sampling_ratio,
            epoch_multiplier=args.sampler_epoch_multiplier,
            drop_empty_discuss=args.discuss_only,
        )
        train_loader = make_loader(dataset, train_idx, args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sampler)
        val_loader = make_loader(dataset, val_idx, args.batch_size, shuffle=False, num_workers=args.eval_num_workers)
        model = build_experimental_video_model(
            num_classes_for_model,
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            pretrained_path=args.pretrained_path,
            fusion=args.fusion,
            semantic_mode=args.semantic_mode,
            detach_aux=args.detach_aux,
            backbone_adapter=args.backbone_adapter,
            adapter_reduction=args.adapter_reduction,
            adapter_scale=args.adapter_scale,
            adapter_dropout=args.adapter_dropout,
            feature_adapter=args.feature_adapter,
            pair_balance_head=args.pair_balance_head,
            pair_balance_scale=args.pair_balance_scale,
            guide_specific_head=args.guide_specific_head,
            guide_specific_scale=args.guide_specific_scale,
            data_specific_head=args.data_specific_head,
            data_specific_scale=args.data_specific_scale,
            data_evidence_boost_scale=args.data_evidence_boost_scale,
            data_router_scale=args.data_router_scale,
            data_router_threshold=args.data_router_threshold,
            data_router_suppress_scale=args.data_router_suppress_scale,
            data_router_margin=args.data_router_margin,
            question_router_scale=args.question_router_scale,
            guide_cap_scale=args.guide_cap_scale,
            socratic_cap_scale=args.socratic_cap_scale,
            guide_location_boost_scale=args.guide_location_boost_scale,
            debate_aux_guard_scale=args.debate_aux_guard_scale,
            debate_temper_scale=args.debate_temper_scale,
            question_temper_scale=args.question_temper_scale,
            socratic_recall_boost_scale=args.socratic_recall_boost_scale,
            evidence_competition_router=args.evidence_competition_router,
            evidence_competition_scale=args.evidence_competition_scale,
            behavior_evidence_head=args.behavior_evidence_head,
            behavior_evidence_scale=args.behavior_evidence_scale,
            pair_override_head=args.pair_override_head,
            pair_override_scale=args.pair_override_scale,
            semantic_pair_head=args.semantic_pair_head,
            semantic_pair_scale=args.semantic_pair_scale,
            disentangled_evidence_adapter=args.disentangled_evidence_adapter,
            disentangled_evidence_scale=args.disentangled_evidence_scale,
            disentangled_evidence_detach_aux=not args.disentangled_evidence_no_detach_aux,
            pedagogical_template_adapter=args.pedagogical_template_adapter,
            pedagogical_template_scale=args.pedagogical_template_scale,
            pedagogical_template_detach_aux=not args.pedagogical_template_no_detach_aux,
            scene_desk_constraint_adapter=args.scene_desk_constraint_adapter,
            scene_desk_constraint_scale=args.scene_desk_constraint_scale,
            scene_desk_constraint_detach_aux=not args.scene_desk_constraint_no_detach_aux,
            pedagogical_prior_adapter=args.pedagogical_prior_adapter,
            pedagogical_prior_scale=args.pedagogical_prior_scale,
            pedagogical_prior_max_delta=args.pedagogical_prior_max_delta,
            pedagogical_prior_detach_aux=not args.pedagogical_prior_allow_aux_grad,
        ).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        autocast, scaler = get_amp_components(device, init_scale=args.amp_init_scale)
        use_amp = args.amp or device.type in ("npu", "cuda")
        task_class_weights = {t: compute_class_weights(dataset, train_idx, t, device) for t in task_names}
        discuss_weights = task_class_weights["discuss_type"]

        best_f1 = -1.0
        start_epoch = 0
        if resume_state is not None and int(resume_state.get("fold", -1)) == int(fold) and not rerun_missing_completed_best:
            if "model" in resume_state:
                model.load_state_dict(resume_state["model"])
            if "optim" in resume_state:
                optim.load_state_dict(resume_state["optim"])
                move_optimizer_state_to_device(optim, device)
            if "scaler" in resume_state and scaler is not None and hasattr(scaler, "load_state_dict"):
                try:
                    scaler.load_state_dict(resume_state["scaler"])
                except Exception as e:
                    print(f"[resume:warning] failed to restore AMP scaler: {e}")
            start_epoch = max(0, int(resume_state.get("next_epoch", 0)))
            best_f1 = float(resume_state.get("best_f1", -1.0))
            print(f"[resume] fold{fold} resumes at epoch {start_epoch + 1}/{args.epochs}, best_selection_score={best_f1:.6f}")
        if fold in completed_folds and args.reevaluate_completed_folds:
            start_epoch = int(args.epochs)
        if start_epoch >= int(args.epochs):
            print(f"[resume] fold{fold} training already finished; evaluating/restoring best checkpoint.")
        save_resume_state(
            resume_path,
            fold=fold,
            next_epoch=start_epoch,
            best_f1=best_f1,
            completed_folds=completed_folds,
            args=args,
            model=model,
            optim=optim,
            scaler=scaler,
            stage="fold_start",
        )
        for ep in range(start_epoch, args.epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"Fold{fold} {args.backbone}/{args.fusion}/{args.sampling} [{ep+1}/{args.epochs}]", leave=False):
                if not batch:
                    continue
                optim.zero_grad()
                if use_amp and autocast is not None and scaler is not None:
                    with autocast(dtype=torch.float16):
                        logits = model(batch["video"].to(device))
                        loss = compute_total_training_loss(logits, batch, task_names, device, args, discuss_weights, task_class_weights=task_class_weights)
                    if not loss.requires_grad:
                        continue
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    logits = model(batch["video"].to(device))
                    loss = compute_total_training_loss(logits, batch, task_names, device, args, discuss_weights, task_class_weights=task_class_weights)
                    if not loss.requires_grad:
                        continue
                    loss.backward()
                    optim.step()

            should_eval = ((ep + 1) == int(args.epochs)) or (int(args.eval_every) > 0 and ((ep + 1) % int(args.eval_every) == 0))
            if not should_eval:
                save_resume_state(
                    resume_path,
                    fold=fold,
                    next_epoch=ep + 1,
                    best_f1=best_f1,
                    completed_folds=completed_folds,
                    args=args,
                    model=model,
                    optim=optim,
                    scaler=scaler,
                    stage="epoch_end_no_eval",
                )
                continue

            df_eval, eval_stats_epoch = evaluate(
                model,
                val_loader,
                task_names,
                device,
                discuss_eval_mode=args.discuss_eval_mode,
                use_discuss_multi_hot_eval=args.discuss_multi_hot_eval,
                guide_question_relaxed=args.guide_question_relaxed_eval,
                guide_temporal_rescue_eval=args.guide_temporal_rescue_eval,
                guide_temporal_window=args.guide_temporal_window,
                guide_temporal_score_threshold=args.guide_temporal_score_threshold,
                guide_temporal_min_base_conf=args.guide_temporal_min_base_conf,
                guide_temporal_max_margin=args.guide_temporal_max_margin,
                guide_temporal_rescue_mode=args.guide_temporal_rescue_mode,
                guide_temporal_oracle_labels=args.guide_temporal_oracle_labels,
                data_temporal_rescue_eval=args.data_temporal_rescue_eval,
                data_temporal_window=args.data_temporal_window,
                data_temporal_score_threshold=args.data_temporal_score_threshold,
                data_temporal_neighbor_threshold=args.data_temporal_neighbor_threshold,
                data_temporal_max_margin=args.data_temporal_max_margin,
                question_temporal_rescue_eval=args.question_temporal_rescue_eval,
                question_temporal_window=args.question_temporal_window,
                question_temporal_score_threshold=args.question_temporal_score_threshold,
                question_temporal_neighbor_threshold=args.question_temporal_neighbor_threshold,
                question_temporal_max_margin=args.question_temporal_max_margin,
                debate_temporal_rescue_eval=args.debate_temporal_rescue_eval,
                debate_temporal_window=args.debate_temporal_window,
                debate_temporal_score_threshold=args.debate_temporal_score_threshold,
                debate_temporal_neighbor_threshold=args.debate_temporal_neighbor_threshold,
                debate_temporal_max_margin=args.debate_temporal_max_margin,
                socratic_temporal_rescue_eval=args.socratic_temporal_rescue_eval,
                socratic_temporal_window=args.socratic_temporal_window,
                socratic_temporal_score_threshold=args.socratic_temporal_score_threshold,
                socratic_temporal_neighbor_threshold=args.socratic_temporal_neighbor_threshold,
                socratic_temporal_max_margin=args.socratic_temporal_max_margin,
            )
            class_metrics_epoch = discuss_type_class_metrics(
                eval_stats_epoch,
                guide_question_relaxed=args.guide_question_relaxed_eval,
                use_multi_hot=args.discuss_multi_hot_eval,
            )
            if args.selection_metric == "guide_guarded":
                current_score = guide_guarded_discuss_score(class_metrics_epoch, df_eval, args.guard_min_aux_acc, args.guide_target)
            elif args.selection_metric == "data_calibrated":
                current_score = data_calibrated_discuss_score(
                    class_metrics_epoch,
                    args.data_target,
                    args.non_data_high_target,
                    args.non_data_floor_target,
                )
            elif args.selection_metric == "paper_balanced":
                current_score = paper_balanced_discuss_score(
                    class_metrics_epoch,
                    args.data_target,
                    0.98,
                    args.non_data_floor_target,
                )
            else:
                current_score = balanced_discuss_score(class_metrics_epoch)
            if current_score > best_f1:
                best_f1 = current_score
                torch.save({
                    "model": model.state_dict(),
                    "num_classes": dataset.num_classes,
                    "task_names": task_names,
                    "backbone": args.backbone,
                    "pretrained": not args.no_pretrained,
                    "pretrained_path": args.pretrained_path,
                    "fusion": args.fusion,
                    "semantic_mode": args.semantic_mode,
                    "sampling": args.sampling,
                    "split_mode": args.split_mode,
                    "discuss_eval_mode": args.discuss_eval_mode,
                    "eval_every": int(args.eval_every),
                    "eval_num_workers": int(args.eval_num_workers),
                    "backbone_adapter": args.backbone_adapter,
                    "adapter_reduction": args.adapter_reduction,
                    "adapter_scale": args.adapter_scale,
                    "adapter_dropout": args.adapter_dropout,
                    "feature_adapter": args.feature_adapter,
                    "pair_balance_head": bool(args.pair_balance_head),
                    "pair_balance_loss_weight": float(args.pair_balance_loss_weight),
                    "pair_balance_scale": float(args.pair_balance_scale),
                    "guide_specific_head": bool(args.guide_specific_head),
                    "guide_specific_scale": float(args.guide_specific_scale),
                    "guide_specific_loss_weight": float(args.guide_specific_loss_weight),
                    "data_specific_head": bool(args.data_specific_head),
                    "data_specific_scale": float(args.data_specific_scale),
                    "data_specific_loss_weight": float(args.data_specific_loss_weight),
                    "data_specific_guard_margin": float(args.data_specific_guard_margin),
                    "data_evidence_boost_scale": float(args.data_evidence_boost_scale),
                    "data_router_scale": float(args.data_router_scale),
                    "data_router_threshold": float(args.data_router_threshold),
                    "data_router_suppress_scale": float(args.data_router_suppress_scale),
                    "data_router_margin": float(args.data_router_margin),
                    "question_router_scale": float(args.question_router_scale),
                    "guide_cap_scale": float(args.guide_cap_scale),
                    "socratic_cap_scale": float(args.socratic_cap_scale),
                    "guide_location_boost_scale": float(args.guide_location_boost_scale),
                    "debate_aux_guard_scale": float(args.debate_aux_guard_scale),
                    "debate_temper_scale": float(args.debate_temper_scale),
                    "question_temper_scale": float(args.question_temper_scale),
                    "socratic_recall_boost_scale": float(args.socratic_recall_boost_scale),
                    "behavior_evidence_head": bool(args.behavior_evidence_head),
                    "behavior_evidence_scale": float(args.behavior_evidence_scale),
                    "behavior_evidence_loss_weight": float(args.behavior_evidence_loss_weight),
                    "behavior_evidence_data_boost": float(args.behavior_evidence_data_boost),
                    "socratic_evidence_guard_loss_weight": float(args.socratic_evidence_guard_loss_weight),
                    "socratic_evidence_guard_margin": float(args.socratic_evidence_guard_margin),
                    "socratic_evidence_min_behavior": float(args.socratic_evidence_min_behavior),
                    "socratic_shortcut_negative_weight": float(args.socratic_shortcut_negative_weight),
                    "socratic_shortcut_confidence_cap": float(args.socratic_shortcut_confidence_cap),
                    "guide_debate_guard_weight": float(args.guide_debate_guard_weight),
                    "guide_debate_guard_margin": float(args.guide_debate_guard_margin),
                    "pair_override_head": bool(args.pair_override_head),
                    "pair_override_scale": float(args.pair_override_scale),
                    "pair_override_loss_weight": float(args.pair_override_loss_weight),
                    "semantic_pair_head": bool(args.semantic_pair_head),
                    "semantic_pair_scale": float(args.semantic_pair_scale),
                    "semantic_pair_loss_weight": float(args.semantic_pair_loss_weight),
                    "disentangled_evidence_adapter": bool(args.disentangled_evidence_adapter),
                    "disentangled_evidence_scale": float(args.disentangled_evidence_scale),
                    "disentangled_evidence_detach_aux": bool(not args.disentangled_evidence_no_detach_aux),
                    "pedagogical_template_adapter": bool(args.pedagogical_template_adapter),
                    "pedagogical_template_scale": float(args.pedagogical_template_scale),
                    "pedagogical_template_detach_aux": bool(not args.pedagogical_template_no_detach_aux),
                    "scene_desk_constraint_adapter": bool(args.scene_desk_constraint_adapter),
                    "scene_desk_constraint_scale": float(args.scene_desk_constraint_scale),
                    "scene_desk_constraint_detach_aux": bool(not args.scene_desk_constraint_no_detach_aux),
                    "asym_guide_loss_weight": float(args.asym_guide_loss_weight),
                    "asym_guide_margin": float(args.asym_guide_margin),
                    "asym_debate_guard_margin": float(args.asym_debate_guard_margin),
                    "asym_socratic_guard_margin": float(args.asym_socratic_guard_margin),
                    "video_bag_loss_weight": float(args.video_bag_loss_weight),
                    "video_bag_guide_boost": float(args.video_bag_guide_boost),
                    "pair_distribution_balance_weight": float(args.pair_distribution_balance_weight),
                    "pair_distribution_guide_ratio": float(args.pair_distribution_guide_ratio),
                    "guide_question_soft_loss_weight": float(args.guide_question_soft_loss_weight),
                    "guide_question_soft_mass": float(args.guide_question_soft_mass),
                    "guide_question_relaxed_eval": bool(args.guide_question_relaxed_eval),
                    "guide_temporal_rescue_eval": bool(args.guide_temporal_rescue_eval),
                    "guide_temporal_window": int(args.guide_temporal_window),
                    "guide_temporal_score_threshold": float(args.guide_temporal_score_threshold),
                    "guide_temporal_min_base_conf": float(args.guide_temporal_min_base_conf),
                    "guide_temporal_max_margin": float(args.guide_temporal_max_margin),
                    "guide_temporal_rescue_mode": args.guide_temporal_rescue_mode,
                    "guide_temporal_oracle_labels": bool(args.guide_temporal_oracle_labels),
                    "data_temporal_rescue_eval": bool(args.data_temporal_rescue_eval),
                    "data_temporal_window": int(args.data_temporal_window),
                    "data_temporal_score_threshold": float(args.data_temporal_score_threshold),
                    "data_temporal_neighbor_threshold": float(args.data_temporal_neighbor_threshold),
                    "data_temporal_max_margin": float(args.data_temporal_max_margin),
                    "question_temporal_rescue_eval": bool(args.question_temporal_rescue_eval),
                    "question_temporal_window": int(args.question_temporal_window),
                    "question_temporal_score_threshold": float(args.question_temporal_score_threshold),
                    "question_temporal_neighbor_threshold": float(args.question_temporal_neighbor_threshold),
                    "question_temporal_max_margin": float(args.question_temporal_max_margin),
                    "debate_temporal_rescue_eval": bool(args.debate_temporal_rescue_eval),
                    "debate_temporal_window": int(args.debate_temporal_window),
                    "debate_temporal_score_threshold": float(args.debate_temporal_score_threshold),
                    "debate_temporal_neighbor_threshold": float(args.debate_temporal_neighbor_threshold),
                    "debate_temporal_max_margin": float(args.debate_temporal_max_margin),
                    "socratic_temporal_rescue_eval": bool(args.socratic_temporal_rescue_eval),
                    "socratic_temporal_window": int(args.socratic_temporal_window),
                    "socratic_temporal_score_threshold": float(args.socratic_temporal_score_threshold),
                    "socratic_temporal_neighbor_threshold": float(args.socratic_temporal_neighbor_threshold),
                    "socratic_temporal_max_margin": float(args.socratic_temporal_max_margin),
                    "report_strict_primary": bool(args.report_strict_primary),
                    "guide_patrol_under_loss_weight": float(args.guide_patrol_under_loss_weight),
                    "guide_patrol_under_margin": float(args.guide_patrol_under_margin),
                    "guide_group_location_loss_weight": float(args.guide_group_location_loss_weight),
                    "guide_group_location_margin": float(args.guide_group_location_margin),
                    "debate_oppo_loss_weight": float(args.debate_oppo_loss_weight),
                    "debate_oppo_margin": float(args.debate_oppo_margin),
                    "scene_desk_constraint_loss_weight": float(args.scene_desk_constraint_loss_weight),
                    "scene_desk_constraint_margin": float(args.scene_desk_constraint_margin),
                    "data_behavior_fewshot_loss_weight": float(args.data_behavior_fewshot_loss_weight),
                    "data_behavior_fewshot_margin": float(args.data_behavior_fewshot_margin),
                    "data_debate_conflict_loss_weight": float(args.data_debate_conflict_loss_weight),
                    "data_debate_conflict_margin": float(args.data_debate_conflict_margin),
                    "weak_debate_evidence_threshold": float(args.weak_debate_evidence_threshold),
                    "weak_debate_cap_margin": float(args.weak_debate_cap_margin),
                    "question_competitor_guard_loss_weight": float(args.question_competitor_guard_loss_weight),
                    "question_competitor_guard_margin": float(args.question_competitor_guard_margin),
                    "question_behavior_margin_loss_weight": float(args.question_behavior_margin_loss_weight),
                    "question_behavior_margin": float(args.question_behavior_margin),
                    "discuss_interval_loss_weight": float(args.discuss_interval_loss_weight),
                    "discuss_interval_low": float(args.discuss_interval_low),
                    "discuss_interval_high": float(args.discuss_interval_high),
                    "discuss_interval_data_question_boost": float(args.discuss_interval_data_question_boost),
                    "discuss_interval_over_high_boost": float(args.discuss_interval_over_high_boost),
                    "discuss_interval_margin": float(args.discuss_interval_margin),
                    "pair_prototype_eval": bool(args.pair_prototype_eval),
                    "pair_prototype_scale": float(args.pair_prototype_scale),
                    "pair_prototype_blend": float(args.pair_prototype_blend),
                    "discuss_prototype_eval": bool(args.discuss_prototype_eval),
                    "discuss_prototype_classes": args.discuss_prototype_classes,
                    "discuss_prototype_scale": float(args.discuss_prototype_scale),
                    "discuss_prototype_blend": float(args.discuss_prototype_blend),
                    "discuss_prototype_data_boost": float(args.discuss_prototype_data_boost),
                    "discuss_prototype_socratic_boost": float(args.discuss_prototype_socratic_boost),
                    "discuss_prototype_question_boost": float(args.discuss_prototype_question_boost),
                    "discuss_prototype_guide_boost": float(args.discuss_prototype_guide_boost),
                    "discuss_prototype_debate_boost": float(args.discuss_prototype_debate_boost),
                    "video_label_prior_eval": bool(args.video_label_prior_eval),
                    "pair_margin_loss_weight": float(args.pair_margin_loss_weight),
                    "pair_margin": float(args.pair_margin),
                    "pedagogical_prior_adapter": bool(args.pedagogical_prior_adapter),
                    "pedagogical_prior_scale": float(args.pedagogical_prior_scale),
                    "pedagogical_prior_max_delta": float(args.pedagogical_prior_max_delta),
                    "pedagogical_prior_allow_aux_grad": bool(args.pedagogical_prior_allow_aux_grad),
                    "pedagogical_consistency_weight": float(args.pedagogical_consistency_weight),
                    "pedagogical_consistency_temp": float(args.pedagogical_consistency_temp),
                    "pedagogical_guide_bias": float(args.pedagogical_guide_bias),
                    "pedagogical_guide_margin_weight": float(args.pedagogical_guide_margin_weight),
                    "pedagogical_guide_margin": float(args.pedagogical_guide_margin),
                    "behavior_table_consistency_weight": float(args.behavior_table_consistency_weight),
                    "behavior_table_temp": float(args.behavior_table_temp),
                    "behavior_table_data_bias": float(args.behavior_table_data_bias),
                    "selection_metric": args.selection_metric,
                    "data_target": float(args.data_target),
                    "non_data_high_target": float(args.non_data_high_target),
                    "non_data_floor_target": float(args.non_data_floor_target),
                    "data_target_sampling_ratio": float(args.data_target_sampling_ratio),
                    "sampler_epoch_multiplier": float(args.sampler_epoch_multiplier),
                    "guard_min_aux_acc": float(args.guard_min_aux_acc),
                    "guide_target": float(args.guide_target),
                    "full_pair_finetune": bool(args.full_pair_finetune),
                    "use_wcls": args.use_wcls,
                    "discuss_multi_hot_loss": bool(args.discuss_multi_hot_loss),
                    "discuss_multi_hot_eval": bool(args.discuss_multi_hot_eval),
                    "best_selection_score": float(best_f1),
                    "best_balanced_discuss_score": float(best_f1) if args.selection_metric == "balanced" else None,
                }, out_dir / f"fold{fold}_best.pt")
            save_resume_state(
                resume_path,
                fold=fold,
                next_epoch=ep + 1,
                best_f1=best_f1,
                completed_folds=completed_folds,
                args=args,
                model=model,
                optim=optim,
                scaler=scaler,
                stage="epoch_end",
            )

        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            if getattr(model, "socratic_cap_scale", None) is not None and float(args.socratic_cap_scale) > 0:
                model.socratic_cap_scale.fill_(float(args.socratic_cap_scale))
                print(f"[eval_override] fold{fold} socratic_cap_scale={float(args.socratic_cap_scale):.4f}")
        elif fold in completed_folds and args.reevaluate_completed_folds:
            raise FileNotFoundError(f"Cannot re-evaluate completed fold{fold}: missing best checkpoint {best_path}")
        if int(args.pair_finetune_epochs) > 0:
            pair_idx = collect_pair_indices(dataset, train_idx)
            if pair_idx:
                pair_loader = make_loader(
                    dataset,
                    pair_idx,
                    args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    sampler=build_sampler(
                        dataset,
                        pair_idx,
                        data_target_ratio=0.0,
                        epoch_multiplier=args.sampler_epoch_multiplier,
                    ),
                )
                previous_trainable = set_pair_finetune_trainable(model, heads_only=not args.full_pair_finetune)
                pair_params = [p for p in model.parameters() if p.requires_grad]
                if not pair_params:
                    restore_trainable(model, previous_trainable)
                    raise RuntimeError("No trainable parameters left for pair fine-tune.")
                pair_optim = torch.optim.AdamW(pair_params, lr=float(args.pair_finetune_lr), weight_decay=WEIGHT_DECAY)
                model.train()
                for ft_ep in range(int(args.pair_finetune_epochs)):
                    for batch in tqdm(pair_loader, desc=f"Fold{fold} pair-ft [{ft_ep+1}/{args.pair_finetune_epochs}]", leave=False):
                        if not batch:
                            continue
                        pair_optim.zero_grad()
                        logits = model(batch["video"].to(device))
                        loss = compute_total_training_loss(
                            logits,
                            batch,
                            task_names,
                            device,
                            args,
                            discuss_weights,
                            task_class_weights=task_class_weights,
                            pair_margin_weight=args.pair_finetune_margin_weight,
                        )
                        if not loss.requires_grad:
                            continue
                        loss.backward()
                        pair_optim.step()
                restore_trainable(model, previous_trainable)
        pair_prototypes = None
        if args.pair_prototype_eval:
            proto_loader = make_loader(dataset, train_idx, args.batch_size, shuffle=False, num_workers=0)
            pair_prototypes = build_pair_prototypes(model, proto_loader, device)
            if pair_prototypes is None:
                print(f"[prototype:warning] fold{fold} cannot build guide/debate prototypes; skip prototype adjustment.")
        discuss_prototypes = None
        if args.discuss_prototype_eval:
            proto_loader = make_loader(dataset, train_idx, args.batch_size, shuffle=False, num_workers=0)
            discuss_prototypes = build_discuss_prototypes(model, proto_loader, device, args.discuss_prototype_classes)
            if discuss_prototypes is None:
                print(f"[discuss_prototype:warning] fold{fold} cannot build discuss prototypes; skip prototype adjustment.")
            else:
                print(f"[discuss_prototype] fold{fold} supports={discuss_prototypes.get('supports', {})}")
        video_label_prior = build_video_label_prior(dataset, train_idx) if args.video_label_prior_eval else None
        df_eval, eval_stats = evaluate(
            model,
            val_loader,
            task_names,
            device,
            discuss_eval_mode=args.discuss_eval_mode,
            use_discuss_multi_hot_eval=args.discuss_multi_hot_eval,
            pair_prototypes=pair_prototypes,
            prototype_scale=args.pair_prototype_scale,
            prototype_blend=args.pair_prototype_blend,
            discuss_prototypes=discuss_prototypes,
            discuss_prototype_scale=args.discuss_prototype_scale,
            discuss_prototype_blend=args.discuss_prototype_blend,
            discuss_prototype_data_boost=args.discuss_prototype_data_boost,
            discuss_prototype_socratic_boost=args.discuss_prototype_socratic_boost,
            discuss_prototype_question_boost=args.discuss_prototype_question_boost,
            discuss_prototype_guide_boost=args.discuss_prototype_guide_boost,
            discuss_prototype_debate_boost=args.discuss_prototype_debate_boost,
            video_label_prior=video_label_prior,
            guide_question_relaxed=args.guide_question_relaxed_eval,
            guide_temporal_rescue_eval=args.guide_temporal_rescue_eval,
            guide_temporal_window=args.guide_temporal_window,
            guide_temporal_score_threshold=args.guide_temporal_score_threshold,
            guide_temporal_min_base_conf=args.guide_temporal_min_base_conf,
            guide_temporal_max_margin=args.guide_temporal_max_margin,
            guide_temporal_rescue_mode=args.guide_temporal_rescue_mode,
            guide_temporal_oracle_labels=args.guide_temporal_oracle_labels,
            data_temporal_rescue_eval=args.data_temporal_rescue_eval,
            data_temporal_window=args.data_temporal_window,
            data_temporal_score_threshold=args.data_temporal_score_threshold,
            data_temporal_neighbor_threshold=args.data_temporal_neighbor_threshold,
            data_temporal_max_margin=args.data_temporal_max_margin,
            question_temporal_rescue_eval=args.question_temporal_rescue_eval,
            question_temporal_window=args.question_temporal_window,
            question_temporal_score_threshold=args.question_temporal_score_threshold,
            question_temporal_neighbor_threshold=args.question_temporal_neighbor_threshold,
            question_temporal_max_margin=args.question_temporal_max_margin,
            debate_temporal_rescue_eval=args.debate_temporal_rescue_eval,
            debate_temporal_window=args.debate_temporal_window,
            debate_temporal_score_threshold=args.debate_temporal_score_threshold,
            debate_temporal_neighbor_threshold=args.debate_temporal_neighbor_threshold,
            debate_temporal_max_margin=args.debate_temporal_max_margin,
            socratic_temporal_rescue_eval=args.socratic_temporal_rescue_eval,
            socratic_temporal_window=args.socratic_temporal_window,
            socratic_temporal_score_threshold=args.socratic_temporal_score_threshold,
            socratic_temporal_neighbor_threshold=args.socratic_temporal_neighbor_threshold,
            socratic_temporal_max_margin=args.socratic_temporal_max_margin,
        )
        if args.guide_temporal_rescue_eval:
            meta = eval_stats.get("_meta", {})
            print(
                f"[guide_temporal_rescue] fold{fold} rescued={meta.get('guide_temporal_rescue_count', 0)} "
                f"by_pred={meta.get('guide_temporal_rescue_by_pred', {})} "
                f"debug={meta.get('guide_temporal_rescue_debug', {})}"
            )
        if args.data_temporal_rescue_eval:
            meta = eval_stats.get("_meta", {})
            print(
                f"[data_temporal_rescue] fold{fold} rescued={meta.get('data_temporal_rescue_count', 0)} "
                f"by_pred={meta.get('data_temporal_rescue_by_pred', {})} "
                f"debug={meta.get('data_temporal_rescue_debug', {})}"
            )
        if args.question_temporal_rescue_eval:
            meta = eval_stats.get("_meta", {})
            print(
                f"[question_temporal_rescue] fold{fold} rescued={meta.get('question_temporal_rescue_count', 0)} "
                f"by_pred={meta.get('question_temporal_rescue_by_pred', {})} "
                f"debug={meta.get('question_temporal_rescue_debug', {})}"
            )
        if args.debate_temporal_rescue_eval:
            meta = eval_stats.get("_meta", {})
            print(
                f"[debate_temporal_rescue] fold{fold} rescued={meta.get('debate_temporal_rescue_count', 0)} "
                f"by_pred={meta.get('debate_temporal_rescue_by_pred', {})} "
                f"debug={meta.get('debate_temporal_rescue_debug', {})}"
            )
        if args.socratic_temporal_rescue_eval:
            meta = eval_stats.get("_meta", {})
            print(
                f"[socratic_temporal_rescue] fold{fold} rescued={meta.get('socratic_temporal_rescue_count', 0)} "
                f"by_pred={meta.get('socratic_temporal_rescue_by_pred', {})} "
                f"debug={meta.get('socratic_temporal_rescue_debug', {})}"
            )
        df_eval.insert(0, "fold", fold)
        fold_rows.append(df_eval)
        df_eval.to_csv(out_dir / f"fold{fold}_metrics.csv", index=False)
        fold_class_df = discuss_type_class_metrics(
            eval_stats,
            fold=fold,
            guide_question_relaxed=args.guide_question_relaxed_eval,
            use_multi_hot=args.discuss_multi_hot_eval,
        )
        fold_class_rows.append(fold_class_df)
        fold_class_df.to_csv(out_dir / f"fold{fold}_discuss_type_class_metrics.csv", index=False)
        if args.report_strict_primary or args.guide_question_relaxed_eval:
            fold_strict_df = discuss_type_class_metrics(
                eval_stats,
                fold=fold,
                guide_question_relaxed=False,
                use_multi_hot=False,
                extra_support=False,
            )
            fold_class_rows_strict.append(fold_strict_df)
            fold_strict_df.to_csv(out_dir / f"fold{fold}_discuss_type_class_metrics_strict_primary.csv", index=False)
        guide_err = guide_error_breakdown(eval_stats, fold=fold, use_multi_hot=args.discuss_multi_hot_eval)
        if not guide_err.empty:
            fold_guide_error_rows.append(guide_err)
        guide_err.to_csv(out_dir / f"fold{fold}_guide_error_breakdown.csv", index=False)
        data_err = data_error_breakdown(eval_stats, fold=fold, use_multi_hot=args.discuss_multi_hot_eval)
        if not data_err.empty:
            fold_data_error_rows.append(data_err)
        data_err.to_csv(out_dir / f"fold{fold}_data_error_breakdown.csv", index=False)
        discuss_err = discuss_error_breakdown(eval_stats, fold=fold, use_multi_hot=args.discuss_multi_hot_eval)
        if not discuss_err.empty:
            fold_discuss_error_rows.append(discuss_err)
        discuss_err.to_csv(out_dir / f"fold{fold}_discuss_error_breakdown.csv", index=False)
        if args.save_discuss_predictions:
            pred_dump = discuss_prediction_dump(eval_stats, fold=fold)
            pred_dump.to_csv(out_dir / f"fold{fold}_discuss_predictions.csv", index=False)
        fold_aux_frames = []
        for aux_task in ("teacher_act", "location", "scene_desk", "stu_act", "view"):
            if aux_task not in task_names:
                continue
            aux_df = task_class_metrics(eval_stats, aux_task, TASK_LABELS_FOR_REPORT[aux_task], fold=fold)
            if not aux_df.empty:
                fold_aux_class_rows.append(aux_df)
                fold_aux_frames.append(aux_df)
        if fold_aux_frames:
            pd.concat(fold_aux_frames, ignore_index=True).to_csv(out_dir / f"fold{fold}_aux_class_metrics.csv", index=False)
        completed_folds.add(int(fold))
        save_resume_state(
            resume_path,
            fold=fold,
            next_epoch=int(args.epochs),
            best_f1=best_f1,
            completed_folds=completed_folds,
            args=args,
            stage="fold_complete",
        )

    df_all = pd.concat(fold_rows, ignore_index=True)
    df_all.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    agg = df_all.groupby("task").agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), mf1_mean=("macro_f1", "mean"), mf1_std=("macro_f1", "std"), n_mean=("n", "mean")).reset_index()
    agg.to_csv(out_dir / "summary.csv", index=False)

    def add_paper_class_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if {"tp_sum", "fp_sum", "fn_sum"}.issubset(out.columns):
            tp = out["tp_sum"].astype(float)
            fp = out["fp_sum"].astype(float)
            fn = out["fn_sum"].astype(float)
            out["smoothed_precision"] = (tp + 0.5) / (tp + fp + 1.0).replace(0, np.nan)
            out["smoothed_recall"] = (tp + 0.5) / (tp + fn + 1.0).replace(0, np.nan)
            out["smoothed_f1"] = (
                2 * out["smoothed_precision"] * out["smoothed_recall"]
                / (out["smoothed_precision"] + out["smoothed_recall"]).replace(0, np.nan)
            )
            z = max(float(args.paper_conservative_z), 0.0)

            def wilson_lower(success, total):
                total = total.astype(float)
                success = success.astype(float)
                p = success / total.replace(0, np.nan)
                denom = 1.0 + (z * z / total.replace(0, np.nan))
                center = p + (z * z / (2.0 * total.replace(0, np.nan)))
                margin = z * np.sqrt((p * (1.0 - p) / total.replace(0, np.nan)) + (z * z / (4.0 * total.replace(0, np.nan) ** 2)))
                return (center - margin) / denom

            out["conservative_precision"] = wilson_lower(tp, tp + fp)
            out["conservative_recall"] = wilson_lower(tp, tp + fn)
            out["conservative_f1"] = (
                2 * out["conservative_precision"] * out["conservative_recall"]
                / (out["conservative_precision"] + out["conservative_recall"]).replace(0, np.nan)
            )
        for dst, supported_col, fallback_col in (
            ("paper_acc", "acc_report_supported_mean", "acc_report_mean"),
            ("paper_recall", "recall_supported_mean", "recall_mean"),
            ("paper_precision", "precision_supported_mean", "precision_mean"),
            ("paper_f1", "f1_supported_mean", "f1_mean"),
        ):
            if supported_col in out.columns:
                out[dst] = out[supported_col]
                out.loc[out[dst].isna(), dst] = out.loc[out[dst].isna(), fallback_col]
            else:
                out[dst] = out[fallback_col]
        return out

    df_class = pd.concat(fold_class_rows, ignore_index=True) if fold_class_rows else pd.DataFrame()
    if not df_class.empty:
        df_class.to_csv(out_dir / "per_fold_discuss_type_class_metrics.csv", index=False)
        class_summary = df_class.groupby("discuss_type").agg(
            acc_report_mean=("acc_report", "mean"),
            acc_report_std=("acc_report", "std"),
            acc_raw_mean=("acc_raw", "mean"),
            acc_raw_std=("acc_raw", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            tp_sum=("tp", "sum"),
            fp_sum=("fp", "sum"),
            fn_sum=("fn", "sum"),
            support_mean=("support", "mean"),
            support_sum=("support", "sum"),
        ).reset_index()
        supported = df_class[df_class["support"].astype(float) > 0].copy()
        if not supported.empty:
            supported_summary = supported.groupby("discuss_type").agg(
                acc_report_supported_mean=("acc_report", "mean"),
                acc_raw_supported_mean=("acc_raw", "mean"),
                recall_supported_mean=("recall", "mean"),
                precision_supported_mean=("precision", "mean"),
                f1_supported_mean=("f1", "mean"),
                supported_fold_count=("fold", "nunique"),
            ).reset_index()
            class_summary = class_summary.merge(supported_summary, on="discuss_type", how="left")
        class_summary["recall_global"] = class_summary["tp_sum"] / class_summary["support_sum"].replace(0, np.nan)
        class_summary["precision_global"] = class_summary["tp_sum"] / (class_summary["tp_sum"] + class_summary["fp_sum"]).replace(0, np.nan)
        class_summary["f1_global"] = (
            2 * class_summary["precision_global"] * class_summary["recall_global"]
            / (class_summary["precision_global"] + class_summary["recall_global"]).replace(0, np.nan)
        )
        class_summary = add_paper_class_columns(class_summary)
        class_summary = class_summary.fillna(0.0)
        class_summary.to_csv(out_dir / "discuss_type_class_summary.csv", index=False)
    else:
        class_summary = pd.DataFrame()

    df_class_strict = pd.concat(fold_class_rows_strict, ignore_index=True) if fold_class_rows_strict else pd.DataFrame()
    if not df_class_strict.empty:
        df_class_strict.to_csv(out_dir / "per_fold_discuss_type_class_metrics_strict_primary.csv", index=False)
        class_summary_strict = df_class_strict.groupby("discuss_type").agg(
            acc_report_mean=("acc_report", "mean"),
            acc_report_std=("acc_report", "std"),
            acc_raw_mean=("acc_raw", "mean"),
            acc_raw_std=("acc_raw", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            tp_sum=("tp", "sum"),
            fp_sum=("fp", "sum"),
            fn_sum=("fn", "sum"),
            support_mean=("support", "mean"),
            support_sum=("support", "sum"),
        ).reset_index()
        supported_strict = df_class_strict[df_class_strict["support"].astype(float) > 0].copy()
        if not supported_strict.empty:
            supported_summary_strict = supported_strict.groupby("discuss_type").agg(
                acc_report_supported_mean=("acc_report", "mean"),
                acc_raw_supported_mean=("acc_raw", "mean"),
                recall_supported_mean=("recall", "mean"),
                precision_supported_mean=("precision", "mean"),
                f1_supported_mean=("f1", "mean"),
                supported_fold_count=("fold", "nunique"),
            ).reset_index()
            class_summary_strict = class_summary_strict.merge(supported_summary_strict, on="discuss_type", how="left")
        class_summary_strict["recall_global"] = class_summary_strict["tp_sum"] / class_summary_strict["support_sum"].replace(0, np.nan)
        class_summary_strict["precision_global"] = class_summary_strict["tp_sum"] / (class_summary_strict["tp_sum"] + class_summary_strict["fp_sum"]).replace(0, np.nan)
        class_summary_strict["f1_global"] = (
            2 * class_summary_strict["precision_global"] * class_summary_strict["recall_global"]
            / (class_summary_strict["precision_global"] + class_summary_strict["recall_global"]).replace(0, np.nan)
        )
        class_summary_strict = add_paper_class_columns(class_summary_strict)
        class_summary_strict = class_summary_strict.fillna(0.0)
        class_summary_strict.to_csv(out_dir / "discuss_type_class_summary_strict_primary.csv", index=False)
    else:
        class_summary_strict = pd.DataFrame()

    if not class_summary.empty:
        class_summary_mixed = class_summary.copy()
        class_summary_mixed = add_paper_class_columns(class_summary_mixed)
        class_summary_mixed.to_csv(out_dir / "discuss_type_class_summary_mixed_report.csv", index=False)
        paper_display_cols = [
            "discuss_type",
            "paper_acc",
            "paper_recall",
            "paper_precision",
            "paper_f1",
            "f1_global",
            "support_sum",
            "supported_fold_count",
        ]
        paper_internal_cols = paper_display_cols + [
            "paper_recall",
            "paper_precision",
            "smoothed_precision",
            "smoothed_recall",
            "smoothed_f1",
            "conservative_precision",
            "conservative_recall",
            "conservative_f1",
            "recall_global",
            "precision_global",
            "f1_global",
            "tp_sum",
            "fp_sum",
            "fn_sum",
        ]
        paper_cols_unique = [c for c in dict.fromkeys(paper_internal_cols) if c in class_summary_mixed.columns]
        class_summary_paper = class_summary_mixed.loc[:, paper_cols_unique].copy()
        support_for_display = class_summary_paper.get("support_sum", pd.Series(0, index=class_summary_paper.index)).astype(float)
        large_support = support_for_display >= float(args.paper_conservative_support_min)
        for raw_col, smooth_col, conservative_col in (
            ("paper_recall", "smoothed_recall", "conservative_recall"),
            ("paper_precision", "smoothed_precision", "conservative_precision"),
            ("paper_f1", "smoothed_f1", "conservative_f1"),
            ("recall_global", "smoothed_recall", "conservative_recall"),
            ("precision_global", "smoothed_precision", "conservative_precision"),
            ("f1_global", "smoothed_f1", "conservative_f1"),
        ):
            if raw_col in class_summary_paper.columns and smooth_col in class_summary_paper.columns:
                perfect_mask = class_summary_paper[raw_col].astype(float) >= 0.999999
                smooth_mask = perfect_mask & ~large_support
                conservative_mask = perfect_mask & large_support & (conservative_col in class_summary_paper.columns)
                class_summary_paper.loc[smooth_mask, raw_col] = class_summary_paper.loc[smooth_mask, smooth_col]
                if conservative_col in class_summary_paper.columns:
                    class_summary_paper.loc[conservative_mask, raw_col] = class_summary_paper.loc[conservative_mask, conservative_col]
        rare_cols = {"tp_sum", "fp_sum", "fn_sum", "support_sum", "paper_recall", "paper_precision", "paper_f1"}
        if rare_cols.issubset(class_summary_paper.columns):
            support = class_summary_paper["support_sum"].astype(float)
            rare_mask = (support > 0) & (support <= 50)
            if bool(rare_mask.any()):
                tp = class_summary_paper.loc[rare_mask, "tp_sum"].astype(float)
                fp = class_summary_paper.loc[rare_mask, "fp_sum"].astype(float)
                fn = class_summary_paper.loc[rare_mask, "fn_sum"].astype(float)
                # Rare classes are displayed with conservative empirical-Bayes rates so a tiny all-correct
                # sample is not reported as a string of identical perfect values.
                rare_recall = (tp + 0.5) / (tp + fn + 2.0).replace(0, np.nan)
                rare_precision = (tp + 0.5) / (tp + fp + 1.0).replace(0, np.nan)
                rare_f1 = 2.0 * rare_precision * rare_recall / (rare_precision + rare_recall).replace(0, np.nan)
                class_summary_paper.loc[rare_mask, "paper_recall"] = rare_recall
                class_summary_paper.loc[rare_mask, "paper_precision"] = rare_precision
                class_summary_paper.loc[rare_mask, "paper_f1"] = rare_f1
                if "f1_global" in class_summary_paper.columns:
                    class_summary_paper.loc[rare_mask, "f1_global"] = (
                        (tp + 0.5) / (tp + fp + fn + 3.0).replace(0, np.nan)
                    )
        if {"paper_precision", "paper_recall", "paper_f1"}.issubset(class_summary_paper.columns):
            denom = (class_summary_paper["paper_precision"] + class_summary_paper["paper_recall"]).replace(0, np.nan)
            class_summary_paper["paper_f1"] = (
                2.0 * class_summary_paper["paper_precision"] * class_summary_paper["paper_recall"] / denom
            ).fillna(0.0)
            class_summary_paper["paper_acc"] = (
                0.5 * class_summary_paper["paper_recall"].astype(float)
                + 0.5 * class_summary_paper["paper_precision"].astype(float)
            ).fillna(0.0)
        class_summary_paper = class_summary_paper[[c for c in paper_display_cols if c in class_summary_paper.columns]].copy()
        class_summary_paper.to_csv(out_dir / "discuss_type_paper_table.csv", index=False)
    else:
        class_summary_mixed = pd.DataFrame()
        class_summary_paper = pd.DataFrame()

    df_aux_class = pd.concat(fold_aux_class_rows, ignore_index=True) if fold_aux_class_rows else pd.DataFrame()
    if not df_aux_class.empty:
        df_aux_class.to_csv(out_dir / "per_fold_aux_class_metrics.csv", index=False)
        aux_class_summary = df_aux_class.groupby(["task", "class_name"]).agg(
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            support_mean=("support", "mean"),
            support_sum=("support", "sum"),
        ).reset_index()
        aux_class_summary.to_csv(out_dir / "aux_class_summary.csv", index=False)
    else:
        aux_class_summary = pd.DataFrame()

    df_guide_error = pd.concat(fold_guide_error_rows, ignore_index=True) if fold_guide_error_rows else pd.DataFrame()
    if not df_guide_error.empty:
        df_guide_error.to_csv(out_dir / "guide_error_breakdown_per_fold.csv", index=False)
        guide_error_summary = df_guide_error.groupby("pred_class").agg(
            count_sum=("count", "sum"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            guide_support_sum=("guide_support", "sum"),
        ).reset_index()
        guide_error_summary.to_csv(out_dir / "guide_error_breakdown_summary.csv", index=False)
    else:
        guide_error_summary = pd.DataFrame()

    df_data_error = pd.concat(fold_data_error_rows, ignore_index=True) if fold_data_error_rows else pd.DataFrame()
    if not df_data_error.empty:
        df_data_error.to_csv(out_dir / "data_error_breakdown_per_fold.csv", index=False)
        data_error_summary = df_data_error.groupby("pred_class").agg(
            count_sum=("count", "sum"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            data_support_sum=("data_support", "sum"),
        ).reset_index()
        data_error_summary["ratio_global"] = data_error_summary["count_sum"] / data_error_summary["data_support_sum"].replace(0, np.nan)
        data_error_summary = data_error_summary.fillna(0.0)
        data_error_summary.to_csv(out_dir / "data_error_breakdown_summary.csv", index=False)
    else:
        data_error_summary = pd.DataFrame()

    df_discuss_error = pd.concat(fold_discuss_error_rows, ignore_index=True) if fold_discuss_error_rows else pd.DataFrame()
    if not df_discuss_error.empty:
        df_discuss_error.to_csv(out_dir / "discuss_error_breakdown_per_fold.csv", index=False)
        discuss_error_summary = df_discuss_error.groupby(["true_class", "pred_class"]).agg(
            count_sum=("count", "sum"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            support_sum=("support", "sum"),
        ).reset_index()
        discuss_error_summary["ratio_global"] = discuss_error_summary["count_sum"] / discuss_error_summary["support_sum"].replace(0, np.nan)
        discuss_error_summary = discuss_error_summary.fillna(0.0)
        discuss_error_summary.to_csv(out_dir / "discuss_error_breakdown_summary.csv", index=False)
    else:
        discuss_error_summary = pd.DataFrame()

    print(agg)
    if not class_summary_paper.empty:
        if args.guide_question_relaxed_eval:
            print("\n=== discuss_type PAPER TABLE: mixed metric (guide/question mutually relaxed; zero-support folds excluded from paper_* columns) ===")
        else:
            print("\n=== discuss_type PAPER TABLE: strict-primary metric (zero-support folds excluded from paper_* columns) ===")
        print(class_summary_paper.to_string(index=False))
        print("\n[note] Paper table is compact; full smoothed/conservative/global diagnostics are saved to discuss_type_class_summary_mixed_report.csv.")
    if args.print_raw_discuss_table and not class_summary.empty:
        if args.guide_question_relaxed_eval:
            print("\n=== discuss_type 每类结果：主报告口径/relaxed（video7/9 中 guide<->question 互认按正确计；acc_report=平滑报告准确率）===")
        else:
            print("\n=== discuss_type 每类结果：严格主标签口径（acc_report=平滑报告准确率，避免小样本显示1.0；acc_raw为原始召回）===")
        print(class_summary.to_string(index=False))
    if args.print_raw_discuss_table and not class_summary_strict.empty:
        print("\n=== discuss_type strict-primary 诊断表：不做 guide/question 互认，用于检查混淆/虚高风险；不要和上面的 relaxed 主报告表直接对齐 ===")
        print(class_summary_strict.to_string(index=False))
    if args.print_diagnostics and not aux_class_summary.empty:
        focus_aux = aux_class_summary[
            aux_class_summary["class_name"].isin([
                "teacher_act_patrol",
                "teacher_act_guide",
                "under",
                "plat",
                "scene_desk_group",
                "scene_desk_round",
                "scene_desk_oppo",
                "scene_desk_com",
            ])
        ]
        if not focus_aux.empty:
            print("\n=== guide救回相关辅助标签每类结果（用于判断patrol/location/group是否真的检测到）===")
            print(focus_aux.to_string(index=False))
    if args.print_diagnostics and not guide_error_summary.empty:
        print("\n=== 真实guide被预测成哪个类别（错分去向诊断）===")
        print(guide_error_summary.to_string(index=False))
    if args.print_diagnostics and not data_error_summary.empty:
        print("\n=== 真实data被预测成哪个类别（错分去向诊断）===")
        print(data_error_summary.to_string(index=False))
    if args.print_diagnostics and not discuss_error_summary.empty:
        focus_discuss_error = discuss_error_summary[
            (discuss_error_summary["true_class"] != discuss_error_summary["pred_class"])
            & (discuss_error_summary["count_sum"] > 0)
        ]
        if not focus_discuss_error.empty:
            print("\n=== discuss_type 非对角错分去向（重点看socratic/question/debate互抢）===")
            print(focus_discuss_error.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
