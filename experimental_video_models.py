# -*- coding: utf-8 -*-
"""扩展视频多任务模型：baseline / 可学习语义融合 BSF / 多 backbone 消融。

该文件独立于 `video_models.py`，用于论文创新点和消融实验。
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from semantic_fusion_modules import (
    LearnableSemanticFusionAttention,
    LearnableSemanticFusionMLP,
    build_semantic_vectors,
)
from backbone_adapters import inject_internal_adapters
from config import (
    DISCUSS_TYPE_LABELS,
    LOCATION_LABELS,
    SCENE_DESK_LABELS,
    SCENE_INTE_LABELS,
    STU_ACT_LABELS,
    TEACHER_ACT_LABELS,
    VIEW_LABELS,
)


_PED_TASK_LABELS = {
    "scene_desk": SCENE_DESK_LABELS,
    "location": LOCATION_LABELS,
    "teacher_act": TEACHER_ACT_LABELS,
    "stu_act": STU_ACT_LABELS,
    "view": VIEW_LABELS,
    "scene_inte": SCENE_INTE_LABELS,
}

_PED_EXPECTED = {
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

_PED_TASK_WEIGHTS = {
    "scene_desk": 0.75,
    "teacher_act": 4.2,
    "stu_act": 2.8,
    "location": 1.6,
    "view": 1.6,
    "scene_inte": 1.0,
}


def _label_idx(labels: list[str], name: str) -> int | None:
    try:
        return labels.index(name)
    except ValueError:
        return None


class GuideSpecificHead(nn.Module):
    """Auxiliary one-vs-rest expert that calibrates only the guide_discuss logit."""

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.2, init_scale: float = 0.03):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.scale_logit = nn.Parameter(torch.logit(torch.tensor(max(min(float(init_scale), 0.95), 1e-4), dtype=torch.float32)))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        guide_logit = self.net(feat).squeeze(1)
        return guide_logit, torch.sigmoid(self.scale_logit) * guide_logit


class DataSpecificHead(nn.Module):
    """Rare-class one-vs-rest expert for data_discuss using visual and semantic evidence."""

    def __init__(self, dim: int, semantic_dim: int = 0, hidden: int = 256, dropout: float = 0.25, init_scale: float = 0.05):
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        input_dim = int(dim) + self.semantic_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.scale_logit = nn.Parameter(torch.logit(torch.tensor(max(min(float(init_scale), 0.95), 1e-4), dtype=torch.float32)))

    def forward(self, feat: torch.Tensor, semantic_evidence: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = feat
        if self.semantic_dim > 0:
            if semantic_evidence is None:
                semantic_evidence = feat.new_zeros(feat.shape[0], self.semantic_dim)
            semantic_evidence = semantic_evidence.to(device=feat.device, dtype=feat.dtype)
            x = torch.cat([feat, semantic_evidence], dim=1)
        data_logit = self.net(x).squeeze(1)
        return data_logit, torch.sigmoid(self.scale_logit) * data_logit


class BehaviorEvidenceDiscussHead(nn.Module):
    """Semantic-only discuss head built from teacher/student/location evidence.

    This head limits direct video-identity memorization: it receives compact
    pedagogical evidence from auxiliary predictions instead of pooled visual
    features, then contributes a calibrated full five-class logit residual.
    """

    def __init__(self, evidence_dim: int, num_classes: int, hidden: int = 160, dropout: float = 0.25, init_scale: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(evidence_dim)),
            nn.Linear(int(evidence_dim), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, int(num_classes)),
        )
        self.scale_logit = nn.Parameter(torch.logit(torch.tensor(max(min(float(init_scale), 0.95), 1e-4), dtype=torch.float32)))

    def forward(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(evidence)
        residual = logits - logits.mean(dim=1, keepdim=True)
        return logits, torch.sigmoid(self.scale_logit) * residual


class GuideDebateBalanceHead(nn.Module):
    """Small symmetric expert for the guide/debate boundary."""

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.2, init_scale: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.scale_logit = nn.Parameter(torch.logit(torch.tensor(max(min(float(init_scale), 0.95), 1e-4), dtype=torch.float32)))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_logits = self.net(feat)
        return pair_logits, torch.sigmoid(self.scale_logit) * pair_logits


class SemanticGuideDebateHead(nn.Module):
    """Guide/debate expert using visual feature plus auxiliary semantic predictions."""

    def __init__(self, feat_dim: int, semantic_dim: int, hidden: int = 256, dropout: float = 0.2, init_scale: float = 0.25):
        super().__init__()
        dim = int(feat_dim) + int(semantic_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.scale_logit = nn.Parameter(torch.logit(torch.tensor(max(min(float(init_scale), 0.95), 1e-4), dtype=torch.float32)))

    def forward(self, feat: torch.Tensor, semantic_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_logits = self.net(torch.cat([feat, semantic_vec], dim=1))
        return pair_logits, torch.sigmoid(self.scale_logit) * pair_logits


class DisentangledGuideDebateEvidenceAdapter(nn.Module):
    """Independent evidence adapters for guide and debate without pairwise logit competition."""

    def __init__(self, init_scale: float = 0.8, detach_aux: bool = True):
        super().__init__()
        self.detach_aux = bool(detach_aux)
        self.guide_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.debate_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.guide_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.debate_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _prob(self, logits: Dict[str, torch.Tensor], task: str, labels: list[str], label: str, like: torch.Tensor) -> torch.Tensor:
        if task not in logits or label not in labels:
            return like.new_zeros(like.shape[0])
        x = logits[task].detach() if self.detach_aux else logits[task]
        idx = labels.index(label)
        if idx >= x.shape[1]:
            return like.new_zeros(like.shape[0])
        return torch.softmax(x, dim=1)[:, idx]

    def forward(self, discuss_logits: torch.Tensor, aux_logits: Dict[str, torch.Tensor], guide_idx: int, debate_idx: int) -> torch.Tensor:
        p_patrol = self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", discuss_logits)
        p_tguide = self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_guide", discuss_logits)
        p_under = self._prob(aux_logits, "location", LOCATION_LABELS, "under", discuss_logits)
        p_group = self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", discuss_logits)
        p_oppo = self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", discuss_logits)
        p_plat = self._prob(aux_logits, "location", LOCATION_LABELS, "plat", discuss_logits)
        p_discuss = self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", discuss_logits)
        guide_evidence = p_group * (0.35 * p_under + 0.30 * p_plat + 0.20 * p_patrol + 0.15 * p_tguide) + 0.20 * p_patrol * p_under
        debate_evidence = p_oppo * (0.75 * p_plat + 0.25 * p_discuss)
        out = discuss_logits.clone()
        out[:, guide_idx] = out[:, guide_idx] + torch.relu(self.guide_scale) * guide_evidence + self.guide_bias
        out[:, debate_idx] = out[:, debate_idx] + torch.relu(self.debate_scale) * debate_evidence + self.debate_bias
        return out


class PedagogicalTemplateAdapter(nn.Module):
    """Five-class teaching-template logit adapter based on scene layout and teacher location/action."""

    def __init__(self, init_scale: float = 0.6, detach_aux: bool = True):
        super().__init__()
        self.detach_aux = bool(detach_aux)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.class_bias = nn.Parameter(torch.zeros(len(DISCUSS_TYPE_LABELS), dtype=torch.float32))

    def _prob(self, logits: Dict[str, torch.Tensor], task: str, labels: list[str], label: str, like: torch.Tensor) -> torch.Tensor:
        if task not in logits or label not in labels:
            return like.new_zeros(like.shape[0])
        x = logits[task].detach() if self.detach_aux else logits[task]
        idx = labels.index(label)
        if idx >= x.shape[1]:
            return like.new_zeros(like.shape[0])
        return torch.softmax(x, dim=1)[:, idx]

    def _collect_evidence(self, discuss_logits: torch.Tensor, aux_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "p_group": self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", discuss_logits),
            "p_oppo": self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", discuss_logits),
            "p_round": self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", discuss_logits),
            "p_com": self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_com", discuss_logits),
            "p_under": self._prob(aux_logits, "location", LOCATION_LABELS, "under", discuss_logits),
            "p_plat": self._prob(aux_logits, "location", LOCATION_LABELS, "plat", discuss_logits),
            "p_patrol": self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", discuss_logits),
            "p_guide": self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_guide", discuss_logits),
            "p_ques": self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", discuss_logits),
            "p_exp": self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_exp", discuss_logits),
            "p_listen": self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_listen", discuss_logits),
            "p_answer": self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", discuss_logits),
            "p_write": self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_write", discuss_logits),
            "p_discuss": self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", discuss_logits),
            "p_stu_listen": self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_listen", discuss_logits),
            "p_mate": self._prob(aux_logits, "view", VIEW_LABELS, "mate", discuss_logits),
            "p_teacher_view": self._prob(aux_logits, "view", VIEW_LABELS, "teacher", discuss_logits),
            "p_inte_group": self._prob(aux_logits, "scene_inte", SCENE_INTE_LABELS, "scene_inte_group", discuss_logits),
            "p_inte_oto": self._prob(aux_logits, "scene_inte", SCENE_INTE_LABELS, "scene_inte_oto", discuss_logits),
        }

    def forward(self, discuss_logits: torch.Tensor, aux_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        ev = self._collect_evidence(discuss_logits, aux_logits)
        p_group, p_oppo, p_round, p_com = ev["p_group"], ev["p_oppo"], ev["p_round"], ev["p_com"]
        p_under, p_plat = ev["p_under"], ev["p_plat"]
        p_patrol, p_guide, p_ques = ev["p_patrol"], ev["p_guide"], ev["p_ques"]
        p_exp, p_listen = ev["p_exp"], ev["p_listen"]
        p_answer, p_write = ev["p_answer"], ev["p_write"]
        p_discuss, p_stu_listen = ev["p_discuss"], ev["p_stu_listen"]
        p_mate, p_teacher_view = ev["p_mate"], ev["p_teacher_view"]
        p_inte_group, p_inte_oto = ev["p_inte_group"], ev["p_inte_oto"]

        template = discuss_logits.new_zeros(discuss_logits.shape[0], len(DISCUSS_TYPE_LABELS))
        question = (0.35 + 0.65 * p_group) * (
            0.28 * p_ques + 0.20 * p_under + 0.18 * p_answer + 0.16 * p_discuss + 0.10 * p_mate + 0.08 * p_inte_group
        )
        guide = (0.35 + 0.65 * p_group) * (
            0.18 * p_under + 0.14 * p_plat + 0.22 * p_guide + 0.16 * p_exp + 0.12 * p_listen
            + 0.10 * p_inte_oto + 0.08 * (0.5 * p_write + 0.5 * p_stu_listen)
        )
        debate = (0.25 + 0.75 * p_oppo) * (
            0.34 * p_plat + 0.20 * p_guide + 0.16 * p_exp + 0.12 * p_listen + 0.12 * p_discuss + 0.06 * p_mate
        )
        socratic = (0.30 + 0.70 * p_round) * (
            0.18 * p_under + 0.12 * p_plat + 0.22 * p_ques + 0.16 * p_guide + 0.14 * p_exp
            + 0.10 * p_answer + 0.08 * p_mate
        )
        data_behavior = (
            0.32 * p_exp + 0.24 * p_guide + 0.18 * p_patrol
            + 0.18 * p_write + 0.06 * p_stu_listen + 0.02 * p_teacher_view
        )
        data = (0.45 + 0.42 * p_com + 0.13 * p_plat) * data_behavior

        anti_question = 0.20 * p_oppo + 0.18 * p_round + 0.16 * p_com
        anti_guide = 0.22 * p_oppo + 0.12 * p_round + 0.16 * p_com
        anti_debate = 0.28 * p_group + 0.16 * p_round + 0.16 * p_com
        anti_socratic = 0.18 * p_group + 0.18 * p_oppo + 0.14 * p_com
        anti_data = 0.16 * p_group + 0.18 * p_oppo + 0.14 * p_round + 0.16 * p_mate + 0.16 * p_discuss

        template[:, DISCUSS_TYPE_LABELS.index("question_discuss")] = question - anti_question
        template[:, DISCUSS_TYPE_LABELS.index("guide_discuss")] = guide - anti_guide
        template[:, DISCUSS_TYPE_LABELS.index("debate_discuss")] = debate - anti_debate
        template[:, DISCUSS_TYPE_LABELS.index("socratic_discuss")] = socratic - anti_socratic
        template[:, DISCUSS_TYPE_LABELS.index("data_discuss")] = 1.25 * data - anti_data
        template = template - template.mean(dim=1, keepdim=True)
        return discuss_logits + torch.relu(self.scale) * template + self.class_bias


class SceneDeskConstraintAdapter(nn.Module):
    """Formal structural prior: scene_desk is a necessary condition for discuss_type."""

    def __init__(self, init_scale: float = 0.8, detach_aux: bool = True):
        super().__init__()
        self.detach_aux = bool(detach_aux)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(len(DISCUSS_TYPE_LABELS), dtype=torch.float32))

    def _prob(self, logits: Dict[str, torch.Tensor], task: str, labels: list[str], label: str, like: torch.Tensor) -> torch.Tensor:
        if task not in logits or label not in labels:
            return like.new_zeros(like.shape[0])
        x = logits[task].detach() if self.detach_aux else logits[task]
        idx = labels.index(label)
        if idx >= x.shape[1]:
            return like.new_zeros(like.shape[0])
        return torch.softmax(x, dim=1)[:, idx]

    def forward(self, discuss_logits: torch.Tensor, aux_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(DISCUSS_TYPE_LABELS) != 5:
            return discuss_logits
        p_group = self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", discuss_logits)
        p_oppo = self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", discuss_logits)
        p_round = self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", discuss_logits)
        p_com = self._prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_com", discuss_logits)
        p_patrol = self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", discuss_logits)
        p_guide = self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_guide", discuss_logits)
        p_ques = self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", discuss_logits)
        p_exp = self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_exp", discuss_logits)
        p_teacher_listen = self._prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_listen", discuss_logits)
        p_under = self._prob(aux_logits, "location", LOCATION_LABELS, "under", discuss_logits)
        p_plat = self._prob(aux_logits, "location", LOCATION_LABELS, "plat", discuss_logits)
        p_write = self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_write", discuss_logits)
        p_listen = self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_listen", discuss_logits)
        p_answer = self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", discuss_logits)
        p_discuss = self._prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", discuss_logits)
        p_mate = self._prob(aux_logits, "view", VIEW_LABELS, "mate", discuss_logits)
        p_teacher_view = self._prob(aux_logits, "view", VIEW_LABELS, "teacher", discuss_logits)
        p_inte_group = self._prob(aux_logits, "scene_inte", SCENE_INTE_LABELS, "scene_inte_group", discuss_logits)
        p_inte_oto = self._prob(aux_logits, "scene_inte", SCENE_INTE_LABELS, "scene_inte_oto", discuss_logits)

        guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
        question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
        debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
        socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
        data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")

        question_evidence = 0.28 * p_ques + 0.20 * p_answer + 0.18 * p_discuss + 0.14 * p_mate + 0.12 * p_under + 0.08 * p_inte_group
        guide_evidence = (
            0.18 * p_guide + 0.14 * p_exp + 0.12 * p_patrol + 0.10 * p_teacher_listen + 0.12 * p_write + 0.08 * p_listen
            + 0.12 * p_under + 0.10 * p_plat + 0.12 * p_inte_oto
        )
        debate_evidence = 0.38 * p_oppo + 0.20 * p_plat + 0.16 * p_discuss + 0.12 * p_answer + 0.08 * p_mate + 0.06 * p_inte_group
        socratic_evidence = 0.38 * p_round + 0.16 * p_under + 0.14 * p_ques + 0.12 * p_guide + 0.10 * p_answer + 0.06 * p_listen + 0.04 * p_teacher_listen
        data_evidence = 0.26 * p_com + 0.10 * p_plat + 0.25 * p_exp + 0.18 * p_guide + 0.13 * p_patrol + 0.06 * p_write + 0.02 * p_teacher_view

        out = discuss_logits.clone()
        scale = torch.relu(self.scale)

        # Soft evidence gates: add support and subtract only competing high-confidence layouts.
        out[:, question_idx] = out[:, question_idx] + scale * (0.55 * p_group + 0.45 * question_evidence - 0.18 * p_oppo - 0.12 * p_com)
        out[:, guide_idx] = out[:, guide_idx] + scale * (0.50 * p_group + 0.48 * guide_evidence - 0.18 * p_oppo - 0.12 * p_com)
        out[:, debate_idx] = out[:, debate_idx] + scale * (0.65 * p_oppo + 0.35 * debate_evidence - 0.20 * p_group - 0.14 * p_round - 0.14 * p_com)
        out[:, socratic_idx] = out[:, socratic_idx] + scale * (0.62 * p_round + 0.38 * socratic_evidence - 0.14 * p_oppo - 0.12 * p_com)
        out[:, data_idx] = out[:, data_idx] + scale * (0.48 * p_com + 0.52 * data_evidence - 0.14 * p_group - 0.18 * p_oppo - 0.12 * p_round - 0.10 * p_discuss)
        return out + self.bias


class DiscussTypeEvidenceBuilder:
    """Utility methods for semantic evidence used by rare-class heads."""

    @staticmethod
    def prob(
        aux_logits: Dict[str, torch.Tensor],
        task: str,
        labels: list[str],
        label: str,
        like: torch.Tensor,
        detach_aux: bool = True,
    ) -> torch.Tensor:
        if task not in aux_logits or label not in labels:
            return like.new_zeros(like.shape[0])
        x = aux_logits[task].detach() if detach_aux else aux_logits[task]
        idx = labels.index(label)
        if idx >= x.shape[1]:
            return like.new_zeros(like.shape[0])
        return torch.softmax(x, dim=1)[:, idx]

    @classmethod
    def data_evidence(cls, aux_logits: Dict[str, torch.Tensor], like: torch.Tensor, detach_aux: bool = True) -> torch.Tensor:
        p_group = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", like, detach_aux)
        p_oppo = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", like, detach_aux)
        p_round = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", like, detach_aux)
        p_com = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_com", like, detach_aux)
        p_plat = cls.prob(aux_logits, "location", LOCATION_LABELS, "plat", like, detach_aux)
        p_under = cls.prob(aux_logits, "location", LOCATION_LABELS, "under", like, detach_aux)
        p_guide = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_guide", like, detach_aux)
        p_exp = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_exp", like, detach_aux)
        p_patrol = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", like, detach_aux)
        p_write = cls.prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_write", like, detach_aux)
        p_listen = cls.prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_listen", like, detach_aux)
        p_discuss = cls.prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux)
        p_teacher_view = cls.prob(aux_logits, "view", VIEW_LABELS, "teacher", like, detach_aux)
        p_mate = cls.prob(aux_logits, "view", VIEW_LABELS, "mate", like, detach_aux)
        p_inte_oto = cls.prob(aux_logits, "scene_inte", SCENE_INTE_LABELS, "scene_inte_oto", like, detach_aux)

        layout = p_com - torch.maximum(torch.maximum(p_group, p_oppo), p_round)
        teacher_behavior = 0.46 * p_exp + 0.34 * p_guide + 0.20 * p_patrol
        student_behavior = 0.66 * p_write + 0.34 * p_listen
        behavior = 0.48 * teacher_behavior + 0.30 * student_behavior + 0.12 * p_teacher_view + 0.10 * p_plat
        anti_interaction = 0.25 * p_mate + 0.25 * p_discuss + 0.20 * p_under + 0.15 * torch.maximum(p_group, p_round)
        return torch.stack([
            p_com,
            p_plat,
            p_guide,
            p_exp,
            p_patrol,
            p_write,
            p_listen,
            p_teacher_view,
            p_inte_oto,
            layout,
            teacher_behavior,
            student_behavior,
            behavior,
            anti_interaction,
        ], dim=1)

    @classmethod
    def competition_scores(cls, aux_logits: Dict[str, torch.Tensor], like: torch.Tensor, detach_aux: bool = True) -> torch.Tensor:
        p_group = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", like, detach_aux)
        p_oppo = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", like, detach_aux)
        p_round = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", like, detach_aux)
        p_com = cls.prob(aux_logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_com", like, detach_aux)
        p_plat = cls.prob(aux_logits, "location", LOCATION_LABELS, "plat", like, detach_aux)
        p_under = cls.prob(aux_logits, "location", LOCATION_LABELS, "under", like, detach_aux)
        p_guide = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_guide", like, detach_aux)
        p_exp = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_exp", like, detach_aux)
        p_patrol = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", like, detach_aux)
        p_ques = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux)
        p_listen_t = cls.prob(aux_logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_listen", like, detach_aux)
        p_write = cls.prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_write", like, detach_aux)
        p_listen = cls.prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_listen", like, detach_aux)
        p_answer = cls.prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux)
        p_discuss = cls.prob(aux_logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux)
        p_mate = cls.prob(aux_logits, "view", VIEW_LABELS, "mate", like, detach_aux)
        p_teacher_view = cls.prob(aux_logits, "view", VIEW_LABELS, "teacher", like, detach_aux)

        non_com_layout = torch.maximum(torch.maximum(p_group, p_round), p_oppo)
        data_teacher = 0.40 * p_exp + 0.32 * p_guide + 0.20 * p_patrol + 0.08 * p_listen_t
        data_student = 0.58 * p_write + 0.42 * p_listen
        data = (
            1.18 * p_com
            + 0.42 * data_teacher
            + 0.30 * data_student
            + 0.12 * p_teacher_view
            + 0.10 * p_plat
            - 0.48 * non_com_layout
            - 0.18 * p_discuss
            - 0.12 * p_under
        )

        guide_behavior = 0.34 * p_guide + 0.26 * p_patrol + 0.18 * p_exp + 0.12 * p_write + 0.10 * p_listen
        guide = (
            0.72 * p_group
            + 0.52 * p_under
            + 0.46 * guide_behavior
            - 0.42 * p_plat
            - 0.20 * p_com
            - 0.12 * p_round
        )

        question_behavior = 0.30 * p_ques + 0.26 * p_answer + 0.24 * p_discuss + 0.10 * p_patrol + 0.10 * p_listen_t
        question = (
            0.70 * p_group
            + 0.48 * p_plat
            + 0.42 * question_behavior
            + 0.08 * p_under
            - 0.18 * p_com
            - 0.10 * p_round
        )

        debate_behavior = 0.30 * p_discuss + 0.24 * p_answer + 0.22 * p_mate + 0.12 * p_ques + 0.12 * p_listen_t
        debate = (
            0.58 * p_oppo
            + 0.24 * p_round
            + 0.50 * debate_behavior
            - 0.26 * p_com
            - 0.12 * p_group
        )

        socratic_behavior = 0.26 * p_answer + 0.22 * p_discuss + 0.20 * p_ques + 0.16 * p_listen + 0.16 * p_listen_t
        socratic = (
            0.78 * p_round
            + 0.40 * socratic_behavior
            + 0.08 * p_under
            - 0.36 * p_oppo
            - 0.14 * p_com
        )

        scores = like.new_zeros(like.shape[0], len(DISCUSS_TYPE_LABELS))
        scores[:, DISCUSS_TYPE_LABELS.index("question_discuss")] = question
        scores[:, DISCUSS_TYPE_LABELS.index("guide_discuss")] = guide
        scores[:, DISCUSS_TYPE_LABELS.index("debate_discuss")] = debate
        scores[:, DISCUSS_TYPE_LABELS.index("socratic_discuss")] = socratic
        scores[:, DISCUSS_TYPE_LABELS.index("data_discuss")] = data
        return scores


class PedagogicalPriorAdapter(nn.Module):
    """Fixed pedagogical knowledge prior added as a small learnable logit adapter."""

    def __init__(
        self,
        num_classes_per_task: Dict[str, int],
        init_scale: float = 0.18,
        max_delta: float = 2.0,
        detach_aux: bool = False,
    ):
        super().__init__()
        self.detach_aux = bool(detach_aux)
        self.max_delta = float(max_delta)
        self.discuss_classes = int(num_classes_per_task.get("discuss_type", 0))
        self.semantic_tasks = [
            task for task in ("scene_desk", "location", "teacher_act", "stu_act", "view", "scene_inte")
            if task in num_classes_per_task and task in _PED_TASK_LABELS
        ]
        self.scale_logit = nn.Parameter(
            torch.logit(torch.tensor(max(min(float(init_scale), 0.95), 1e-4), dtype=torch.float32))
        )
        self.class_bias = nn.Parameter(torch.zeros(self.discuss_classes, dtype=torch.float32))

        for task in self.semantic_tasks:
            labels = _PED_TASK_LABELS[task]
            n_task = int(num_classes_per_task[task])
            mask = torch.zeros(self.discuss_classes, n_task, dtype=torch.float32)
            for class_idx, discuss_name in enumerate(DISCUSS_TYPE_LABELS[: self.discuss_classes]):
                for label_name in _PED_EXPECTED.get(discuss_name, {}).get(task, set()):
                    label_index = _label_idx(labels, label_name)
                    if label_index is not None and label_index < n_task:
                        mask[class_idx, label_index] = 1.0
            self.register_buffer(f"{task}_mask", mask)

    def _probs(self, aux_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        probs = {}
        for task in self.semantic_tasks:
            if task not in aux_logits:
                continue
            x = aux_logits[task]
            probs[task] = torch.softmax(x.detach() if self.detach_aux else x, dim=1)
        return probs

    def _take(self, probs: Dict[str, torch.Tensor], task: str, label_name: str):
        if task not in probs:
            return None
        label_index = _label_idx(_PED_TASK_LABELS[task], label_name)
        if label_index is None or label_index >= probs[task].shape[1]:
            return None
        return probs[task][:, label_index]

    def _or_zero(self, probs: Dict[str, torch.Tensor], task: str, label_name: str, like: torch.Tensor) -> torch.Tensor:
        value = self._take(probs, task, label_name)
        return value if value is not None else like.new_zeros(like.shape[0])

    def _interaction_scores(self, probs: Dict[str, torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        scores = like.new_zeros(like.shape[0], self.discuss_classes)
        if self.discuss_classes != len(DISCUSS_TYPE_LABELS):
            return scores

        desk_group = self._or_zero(probs, "scene_desk", "scene_desk_group", like)
        desk_oppo = self._or_zero(probs, "scene_desk", "scene_desk_oppo", like)
        desk_round = self._or_zero(probs, "scene_desk", "scene_desk_round", like)
        desk_com = self._or_zero(probs, "scene_desk", "scene_desk_com", like)
        t_guide = self._or_zero(probs, "teacher_act", "teacher_act_guide", like)
        t_exp = self._or_zero(probs, "teacher_act", "teacher_act_exp", like)
        t_listen = self._or_zero(probs, "teacher_act", "teacher_act_listen", like)
        t_ques = self._or_zero(probs, "teacher_act", "teacher_act_ques", like)
        t_patrol = self._or_zero(probs, "teacher_act", "teacher_act_patrol", like)
        loc_under = self._or_zero(probs, "location", "under", like)
        loc_plat = self._or_zero(probs, "location", "plat", like)
        s_answer = self._or_zero(probs, "stu_act", "stu_act_answer", like)
        s_write = self._or_zero(probs, "stu_act", "stu_act_write", like)
        s_discuss = self._or_zero(probs, "stu_act", "stu_act_discuss", like)
        s_listen = self._or_zero(probs, "stu_act", "stu_act_listen", like)
        v_mate = self._or_zero(probs, "view", "mate", like)
        v_teacher = self._or_zero(probs, "view", "teacher", like)

        guide_evidence = (0.45 + 0.55 * desk_group) * (0.34 * t_guide + 0.20 * t_exp + 0.18 * t_listen + 0.14 * t_patrol + 0.14 * loc_plat) * (
            0.34 * s_write + 0.26 * s_listen + 0.20 * s_answer + 0.20 * s_discuss
        ) * (0.50 + 0.25 * loc_under + 0.25 * loc_plat)
        question_evidence = (0.45 + 0.55 * desk_group) * (0.62 * t_ques + 0.24 * t_patrol + 0.14 * t_guide) * (
            0.58 * s_answer + 0.34 * s_discuss + 0.08 * s_write
        ) * (0.65 * v_mate + 0.35)
        debate_evidence = (0.45 + 0.55 * desk_oppo) * (0.45 * t_guide + 0.30 * t_exp + 0.25 * t_listen) * (
            0.58 * s_discuss + 0.42 * s_answer
        ) * (0.60 + 0.40 * loc_plat)
        socratic_evidence = (0.45 + 0.55 * desk_round) * (0.45 * t_ques + 0.30 * t_listen + 0.25 * t_guide) * (
            0.45 * s_answer + 0.35 * s_discuss + 0.20 * s_listen
        )
        data_teacher = 0.46 * t_exp + 0.34 * t_guide + 0.20 * t_patrol
        data_student = 0.66 * s_write + 0.34 * s_listen
        data_evidence = (
            0.44 * data_teacher
            + 0.25 * data_student
            + 0.12 * v_teacher
            + 0.10 * loc_plat
            + 0.18 * desk_com
        )
        data_counter = 0.18 * torch.maximum(torch.maximum(desk_group, desk_oppo), desk_round) + 0.20 * v_mate + 0.16 * s_discuss

        scores[:, DISCUSS_TYPE_LABELS.index("guide_discuss")] += 0.85 * guide_evidence
        scores[:, DISCUSS_TYPE_LABELS.index("question_discuss")] += 0.85 * question_evidence
        scores[:, DISCUSS_TYPE_LABELS.index("debate_discuss")] += 1.00 * debate_evidence
        scores[:, DISCUSS_TYPE_LABELS.index("socratic_discuss")] += 0.95 * socratic_evidence
        scores[:, DISCUSS_TYPE_LABELS.index("data_discuss")] += 1.65 * data_evidence - data_counter
        return scores

    def forward(self, discuss_logits: torch.Tensor, aux_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.discuss_classes <= 0:
            return discuss_logits
        probs = self._probs(aux_logits)
        prior = discuss_logits.new_zeros(discuss_logits.shape[0], self.discuss_classes)
        normalizer = 0.0
        for task, task_probs in probs.items():
            mask = getattr(self, f"{task}_mask").to(device=task_probs.device, dtype=task_probs.dtype)
            weight = float(_PED_TASK_WEIGHTS.get(task, 1.0))
            prior = prior + weight * (task_probs @ mask.t())
            normalizer += weight
        if normalizer > 0:
            prior = prior / normalizer
        prior = prior + self._interaction_scores(probs, discuss_logits)
        prior = prior - prior.mean(dim=1, keepdim=True)
        delta = torch.sigmoid(self.scale_logit) * self.max_delta * prior
        return discuss_logits + delta + self.class_bias


class MultiScaleLargeKernelAdapter(nn.Module):
    """Lightweight multi-scale large-receptive-field adapter on pooled video features."""

    def __init__(self, dim: int, reduction: int = 4, scale: float = 0.1, dropout: float = 0.0):
        super().__init__()
        hidden = max(32, int(dim) // max(int(reduction), 1))
        self.norm = nn.LayerNorm(dim)
        self.local = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.mid = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.global_branch = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.mix_gate = nn.Sequential(nn.Linear(dim, 3), nn.Softmax(dim=1))
        self.channel_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.norm(feat)
        branches = torch.stack([self.local(x), self.mid(x), self.global_branch(x)], dim=1)
        weights = self.mix_gate(x).unsqueeze(-1)
        mixed = (branches * weights).sum(dim=1)
        return feat + self.scale * self.channel_gate(x) * mixed


class RareBehaviorAdapter(nn.Module):
    """Feature-level residual adapter for rare behavior-defined classes such as data_discuss."""

    def __init__(self, dim: int, reduction: int = 4, scale: float = 0.08, dropout: float = 0.0):
        super().__init__()
        hidden = max(32, int(dim) // max(int(reduction), 1))
        self.norm = nn.LayerNorm(dim)
        self.behavior_branch = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.rare_gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.norm(feat)
        return feat + torch.relu(self.scale) * self.rare_gate(x) * self.behavior_branch(x)


class ExperimentalVideoMultiTaskModel(nn.Module):
    """共享视频 backbone + 多任务 head + 可学习 discuss_type 语义融合模块。"""

    def __init__(
        self,
        num_classes_per_task: Dict[str, int],
        backbone: str = "swin3d_t",
        pretrained: bool = True,
        pretrained_path: str = "",
        dropout: float = 0.3,
        fusion: str = "none",
        semantic_mode: str = "prob",
        detach_aux: bool = False,
        backbone_adapter: str = "none",
        adapter_reduction: int = 4,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.0,
        feature_adapter: str = "none",
        pair_balance_head: bool = False,
        pair_balance_scale: float = 0.05,
        guide_specific_head: bool = False,
        guide_specific_scale: float = 0.05,
        data_specific_head: bool = False,
        data_specific_scale: float = 0.05,
        data_evidence_boost_scale: float = 0.0,
        data_router_scale: float = 0.0,
        data_router_threshold: float = 0.45,
        data_router_suppress_scale: float = 0.0,
        data_router_margin: float = 0.0,
        question_router_scale: float = 0.0,
        guide_cap_scale: float = 0.0,
        socratic_cap_scale: float = 0.0,
        guide_location_boost_scale: float = 0.0,
        debate_aux_guard_scale: float = 0.0,
        debate_temper_scale: float = 0.0,
        question_temper_scale: float = 0.0,
        socratic_recall_boost_scale: float = 0.0,
        evidence_competition_router: bool = False,
        evidence_competition_scale: float = 0.0,
        behavior_evidence_head: bool = False,
        behavior_evidence_scale: float = 0.25,
        pair_override_head: bool = False,
        pair_override_scale: float = 0.25,
        semantic_pair_head: bool = False,
        semantic_pair_scale: float = 0.25,
        disentangled_evidence_adapter: bool = False,
        disentangled_evidence_scale: float = 0.8,
        disentangled_evidence_detach_aux: bool = True,
        pedagogical_template_adapter: bool = False,
        pedagogical_template_scale: float = 0.6,
        pedagogical_template_detach_aux: bool = True,
        scene_desk_constraint_adapter: bool = False,
        scene_desk_constraint_scale: float = 0.8,
        scene_desk_constraint_detach_aux: bool = True,
        pedagogical_prior_adapter: bool = False,
        pedagogical_prior_scale: float = 0.18,
        pedagogical_prior_max_delta: float = 2.0,
        pedagogical_prior_detach_aux: bool = True,
    ):
        super().__init__()
        self.task_names = list(num_classes_per_task.keys())
        self.num_classes_per_task = dict(num_classes_per_task)
        self.fusion = str(fusion).lower()
        self.semantic_mode = str(semantic_mode).lower()
        self.detach_aux = bool(detach_aux)
        self.semantic_tasks = [
            name for name in ("scene_desk", "scene_method", "scene_inte", "teacher_act", "location", "stu_act", "view")
            if name in self.num_classes_per_task
        ]

        self.backbone, feat_dim = build_feature_backbone(backbone, pretrained=pretrained, pretrained_path=pretrained_path)
        self.backbone, self.adapter_blocks = inject_internal_adapters(
            self.backbone,
            adapter=backbone_adapter,
            reduction=adapter_reduction,
            scale=adapter_scale,
            dropout=adapter_dropout,
        )
        feature_adapter_name = str(feature_adapter).lower()
        adapters = []
        if feature_adapter_name in ("ms_lka", "ms_lka_rare"):
            adapters.append(MultiScaleLargeKernelAdapter(feat_dim, reduction=adapter_reduction, scale=adapter_scale, dropout=adapter_dropout))
        if feature_adapter_name in ("rare_behavior", "ms_lka_rare"):
            adapters.append(RareBehaviorAdapter(feat_dim, reduction=adapter_reduction, scale=max(0.04, adapter_scale * 0.8), dropout=adapter_dropout))
        self.feature_adapter = nn.Sequential(*adapters) if adapters else nn.Identity()
        discuss_classes = int(self.num_classes_per_task.get("discuss_type", 0))
        self.pair_balance_head = GuideDebateBalanceHead(feat_dim, hidden=max(128, min(512, feat_dim // 2)), dropout=dropout, init_scale=pair_balance_scale) if bool(pair_balance_head) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.guide_specific_head = GuideSpecificHead(feat_dim, hidden=max(128, min(512, feat_dim // 2)), dropout=dropout, init_scale=guide_specific_scale) if bool(guide_specific_head) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.data_specific_head = DataSpecificHead(feat_dim, semantic_dim=14, hidden=max(128, min(512, feat_dim // 2)), dropout=dropout, init_scale=data_specific_scale) if bool(data_specific_head) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(data_evidence_boost_scale) > 0:
            self.register_buffer("data_evidence_boost_scale", torch.tensor(float(data_evidence_boost_scale), dtype=torch.float32))
        else:
            self.data_evidence_boost_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(data_router_scale) > 0:
            self.register_buffer("data_router_scale", torch.tensor(float(data_router_scale), dtype=torch.float32))
            self.register_buffer("data_router_threshold", torch.tensor(float(data_router_threshold), dtype=torch.float32))
            self.register_buffer("data_router_suppress_scale", torch.tensor(float(data_router_suppress_scale), dtype=torch.float32))
            self.register_buffer("data_router_margin", torch.tensor(float(data_router_margin), dtype=torch.float32))
        else:
            self.data_router_scale = None
            self.data_router_threshold = None
            self.data_router_suppress_scale = None
            self.data_router_margin = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(question_router_scale) > 0:
            self.register_buffer("question_router_scale", torch.tensor(float(question_router_scale), dtype=torch.float32))
        else:
            self.question_router_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(guide_cap_scale) > 0:
            self.register_buffer("guide_cap_scale", torch.tensor(float(guide_cap_scale), dtype=torch.float32))
        else:
            self.guide_cap_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(socratic_cap_scale) > 0:
            self.register_buffer("socratic_cap_scale", torch.tensor(float(socratic_cap_scale), dtype=torch.float32))
        else:
            self.socratic_cap_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(guide_location_boost_scale) > 0:
            self.register_buffer("guide_location_boost_scale", torch.tensor(float(guide_location_boost_scale), dtype=torch.float32))
        else:
            self.guide_location_boost_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(debate_aux_guard_scale) > 0:
            self.register_buffer("debate_aux_guard_scale", torch.tensor(float(debate_aux_guard_scale), dtype=torch.float32))
        else:
            self.debate_aux_guard_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(debate_temper_scale) > 0:
            self.register_buffer("debate_temper_scale", torch.tensor(float(debate_temper_scale), dtype=torch.float32))
        else:
            self.debate_temper_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(question_temper_scale) > 0:
            self.register_buffer("question_temper_scale", torch.tensor(float(question_temper_scale), dtype=torch.float32))
        else:
            self.question_temper_scale = None
        if discuss_classes == len(DISCUSS_TYPE_LABELS) and float(socratic_recall_boost_scale) > 0:
            self.register_buffer("socratic_recall_boost_scale", torch.tensor(float(socratic_recall_boost_scale), dtype=torch.float32))
        else:
            self.socratic_recall_boost_scale = None
        if bool(evidence_competition_router) and discuss_classes == len(DISCUSS_TYPE_LABELS) and float(evidence_competition_scale) > 0:
            self.register_buffer("evidence_competition_scale", torch.tensor(float(evidence_competition_scale), dtype=torch.float32))
        else:
            self.evidence_competition_scale = None
        self.behavior_evidence_head = BehaviorEvidenceDiscussHead(14, discuss_classes, hidden=max(128, min(320, feat_dim // 4)), dropout=dropout, init_scale=behavior_evidence_scale) if bool(behavior_evidence_head) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.pair_override_head = GuideDebateBalanceHead(feat_dim, hidden=max(128, min(512, feat_dim // 2)), dropout=dropout, init_scale=pair_override_scale) if bool(pair_override_head) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.pedagogical_prior_adapter = PedagogicalPriorAdapter(
            num_classes_per_task=self.num_classes_per_task,
            init_scale=pedagogical_prior_scale,
            max_delta=pedagogical_prior_max_delta,
            detach_aux=pedagogical_prior_detach_aux,
        ) if bool(pedagogical_prior_adapter) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.disentangled_evidence_adapter = DisentangledGuideDebateEvidenceAdapter(
            init_scale=disentangled_evidence_scale,
            detach_aux=disentangled_evidence_detach_aux,
        ) if bool(disentangled_evidence_adapter) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.pedagogical_template_adapter = PedagogicalTemplateAdapter(
            init_scale=pedagogical_template_scale,
            detach_aux=pedagogical_template_detach_aux,
        ) if bool(pedagogical_template_adapter) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.scene_desk_constraint_adapter = SceneDeskConstraintAdapter(
            init_scale=scene_desk_constraint_scale,
            detach_aux=scene_desk_constraint_detach_aux,
        ) if bool(scene_desk_constraint_adapter) and discuss_classes == len(DISCUSS_TYPE_LABELS) else None
        self.guide_idx = DISCUSS_TYPE_LABELS.index("guide_discuss")
        self.question_idx = DISCUSS_TYPE_LABELS.index("question_discuss")
        self.debate_idx = DISCUSS_TYPE_LABELS.index("debate_discuss")
        self.socratic_idx = DISCUSS_TYPE_LABELS.index("socratic_discuss")
        self.data_idx = DISCUSS_TYPE_LABELS.index("data_discuss")
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(feat_dim, n) for name, n in self.num_classes_per_task.items()})

        semantic_dims = [self.num_classes_per_task[name] for name in self.semantic_tasks]
        if self.semantic_mode == "both":
            semantic_dims = [d * 2 for d in semantic_dims]
        semantic_dim = sum(semantic_dims)
        self.semantic_pair_head = SemanticGuideDebateHead(
            feat_dim=feat_dim,
            semantic_dim=semantic_dim,
            hidden=max(128, min(512, feat_dim // 2)),
            dropout=dropout,
            init_scale=semantic_pair_scale,
        ) if bool(semantic_pair_head) and discuss_classes == len(DISCUSS_TYPE_LABELS) and semantic_dim > 0 else None
        discuss_classes = int(self.num_classes_per_task.get("discuss_type", 0))

        if self.fusion in ("none", "linear") or discuss_classes <= 0:
            self.fusion_head = None
        elif self.fusion == "mlp":
            self.fusion_head = LearnableSemanticFusionMLP(
                feat_dim=feat_dim,
                semantic_dim=semantic_dim,
                num_classes=discuss_classes,
                hidden_dim=max(256, min(1024, feat_dim)),
                dropout=dropout,
                use_feature=True,
            )
        elif self.fusion == "attn":
            self.fusion_head = LearnableSemanticFusionAttention(
                feat_dim=feat_dim,
                task_dims=semantic_dims,
                num_classes=discuss_classes,
                token_dim=128,
                num_heads=4,
                dropout=dropout,
                use_feature=True,
            )
        else:
            raise ValueError(f"unsupported fusion: {fusion}")

    def extract_features(self, video: torch.Tensor, apply_dropout: bool = False) -> torch.Tensor:
        feat = self.feature_adapter(self.backbone(video))
        return self.dropout(feat) if apply_dropout else feat

    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.extract_features(video, apply_dropout=True)
        logits = {name: self.heads[name](feat) for name in self.task_names}
        if self.fusion_head is not None and self.semantic_tasks:
            semantic_list = build_semantic_vectors(
                logits,
                task_names=self.semantic_tasks,
                mode=self.semantic_mode,
                detach_aux=self.detach_aux,
            )
            if self.fusion == "mlp":
                semantic_vec = torch.cat(semantic_list, dim=1)
                logits["discuss_type"] = self.fusion_head(feat, semantic_vec)
            elif self.fusion == "attn":
                logits["discuss_type"] = self.fusion_head(feat, semantic_list)
        if self.pedagogical_prior_adapter is not None and "discuss_type" in logits:
            logits["discuss_type"] = self.pedagogical_prior_adapter(logits["discuss_type"], logits)
        if self.disentangled_evidence_adapter is not None and "discuss_type" in logits:
            logits["discuss_type"] = self.disentangled_evidence_adapter(logits["discuss_type"], logits, self.guide_idx, self.debate_idx)
        if self.pedagogical_template_adapter is not None and "discuss_type" in logits:
            logits["discuss_type"] = self.pedagogical_template_adapter(logits["discuss_type"], logits)
        if self.scene_desk_constraint_adapter is not None and "discuss_type" in logits:
            logits["discuss_type"] = self.scene_desk_constraint_adapter(logits["discuss_type"], logits)
        if self.guide_location_boost_scale is not None and "discuss_type" in logits:
            like = logits["discuss_type"]
            p_under = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "under", like, detach_aux=True)
            p_plat = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "plat", like, detach_aux=True)
            p_group = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", like, detach_aux=True)
            p_guide = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_guide", like, detach_aux=True)
            p_patrol = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", like, detach_aux=True)
            p_exp = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_exp", like, detach_aux=True)
            p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
            p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
            p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
            p_write = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_write", like, detach_aux=True)
            guide_score = p_under * (0.22 + 0.22 * p_guide + 0.18 * p_patrol + 0.12 * p_exp + 0.10 * p_write + 0.06 * p_group)
            question_score = p_plat * (0.38 + 0.32 * p_ques + 0.26 * p_answer + 0.22 * p_discuss + 0.12 * p_group)
            scale = self.guide_location_boost_scale.to(device=like.device, dtype=like.dtype)
            discuss = like.clone()
            discuss[:, self.guide_idx] = discuss[:, self.guide_idx] + scale * (0.55 * guide_score - 0.34 * question_score)
            discuss[:, self.question_idx] = discuss[:, self.question_idx] + scale * (1.45 * question_score - 0.05 * guide_score)
            logits["discuss_type"] = discuss
        if self.question_router_scale is not None and "discuss_type" in logits:
            like = logits["discuss_type"]
            p_plat = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "plat", like, detach_aux=True)
            p_group = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", like, detach_aux=True)
            p_round = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", like, detach_aux=True)
            p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
            p_patrol = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", like, detach_aux=True)
            p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
            p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
            q_score = (0.20 + 0.80 * p_plat) * (
                0.34 * p_ques + 0.28 * p_answer + 0.24 * p_discuss + 0.12 * p_patrol + 0.12 * p_group
            )
            q_score = torch.relu(q_score - 0.12 * p_round)
            scale = self.question_router_scale.to(device=like.device, dtype=like.dtype)
            discuss = like.clone()
            discuss[:, self.question_idx] = discuss[:, self.question_idx] + scale * q_score
            discuss[:, self.guide_idx] = discuss[:, self.guide_idx] - 0.32 * scale * q_score
            discuss[:, self.socratic_idx] = discuss[:, self.socratic_idx] - 0.16 * scale * q_score
            logits["discuss_type"] = discuss
        if self.guide_cap_scale is not None and "discuss_type" in logits:
            like = logits["discuss_type"]
            p_plat = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "plat", like, detach_aux=True)
            p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
            p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
            p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
            question_like = p_plat * (0.36 * p_ques + 0.30 * p_answer + 0.24 * p_discuss + 0.10)
            scale = self.guide_cap_scale.to(device=like.device, dtype=like.dtype)
            discuss = like.clone()
            discuss[:, self.guide_idx] = discuss[:, self.guide_idx] - scale * question_like
            logits["discuss_type"] = discuss
        if self.pair_balance_head is not None and self.disentangled_evidence_adapter is None and "discuss_type" in logits:
            pair_logits, pair_delta = self.pair_balance_head(feat)
            logits["guide_debate_balance"] = pair_logits
            discuss = logits["discuss_type"].clone()
            discuss[:, self.guide_idx] = discuss[:, self.guide_idx] + pair_delta[:, 0]
            discuss[:, self.debate_idx] = discuss[:, self.debate_idx] + pair_delta[:, 1]
            logits["discuss_type"] = discuss
        if self.guide_specific_head is not None and "discuss_type" in logits:
            guide_binary_logit, guide_delta = self.guide_specific_head(feat)
            logits["guide_specific"] = guide_binary_logit
            discuss = logits["discuss_type"].clone()
            discuss[:, self.guide_idx] = discuss[:, self.guide_idx] + guide_delta
            logits["discuss_type"] = discuss
        if self.data_specific_head is not None and "discuss_type" in logits:
            data_semantic = DiscussTypeEvidenceBuilder.data_evidence(logits, logits["discuss_type"], detach_aux=True)
            data_binary_logit, data_delta = self.data_specific_head(feat, data_semantic)
            logits["data_specific"] = data_binary_logit
            logits["data_specific_evidence"] = data_semantic
            discuss = logits["discuss_type"].clone()
            discuss[:, self.data_idx] = discuss[:, self.data_idx] + data_delta
            logits["discuss_type"] = discuss
        if self.data_evidence_boost_scale is not None and "discuss_type" in logits:
            data_semantic = DiscussTypeEvidenceBuilder.data_evidence(logits, logits["discuss_type"], detach_aux=True)
            p_com = data_semantic[:, 0]
            layout = data_semantic[:, 9]
            teacher_behavior = data_semantic[:, 10]
            student_behavior = data_semantic[:, 11]
            behavior = data_semantic[:, 12]
            anti_interaction = data_semantic[:, 13]
            data_score = (
                0.62 * p_com
                + 0.16 * torch.relu(layout)
                + 0.36 * teacher_behavior
                + 0.20 * student_behavior
                + 0.14 * behavior
                - 0.18 * anti_interaction
            )
            data_score = torch.relu(data_score)
            scale = torch.relu(self.data_evidence_boost_scale)
            discuss = logits["discuss_type"].clone()
            discuss[:, self.data_idx] = discuss[:, self.data_idx] + scale * data_score
            suppress = 0.045 * scale * data_score
            for idx in (self.question_idx, self.guide_idx, self.debate_idx):
                discuss[:, idx] = discuss[:, idx] - suppress
            logits["discuss_type"] = discuss
            logits["data_evidence_boost"] = data_score
        if self.data_router_scale is not None and "discuss_type" in logits:
            data_semantic = DiscussTypeEvidenceBuilder.data_evidence(logits, logits["discuss_type"], detach_aux=True)
            p_com = data_semantic[:, 0]
            layout = data_semantic[:, 9]
            teacher_behavior = data_semantic[:, 10]
            student_behavior = data_semantic[:, 11]
            anti_interaction = data_semantic[:, 13]
            if "scene_desk" in logits and "scene_desk_com" in SCENE_DESK_LABELS:
                desk_logits = logits["scene_desk"].detach()
                com_idx = SCENE_DESK_LABELS.index("scene_desk_com")
                other_desk = torch.cat([desk_logits[:, :com_idx], desk_logits[:, com_idx + 1:]], dim=1)
                com_margin_gate = torch.sigmoid(desk_logits[:, com_idx] - other_desk.max(dim=1).values)
            else:
                com_margin_gate = p_com
            if "data_specific" in logits:
                expert_gate = torch.sigmoid(logits["data_specific"].detach()).to(device=p_com.device, dtype=p_com.dtype)
            else:
                expert_gate = p_com.new_zeros(p_com.shape)
            threshold = self.data_router_threshold.to(device=p_com.device, dtype=p_com.dtype)
            scale = self.data_router_scale.to(device=p_com.device, dtype=p_com.dtype)
            suppress_scale = self.data_router_suppress_scale.to(device=p_com.device, dtype=p_com.dtype)
            com_evidence = torch.maximum(torch.maximum(p_com, com_margin_gate), expert_gate)
            positive_gate = torch.relu(com_evidence - threshold)
            negative_gate = torch.relu(threshold - com_evidence)
            route_score = positive_gate * (
                1.35
                + 0.50 * teacher_behavior
                + 0.38 * student_behavior
                + 0.30 * torch.relu(layout)
                + 0.28 * com_margin_gate
                + 0.36 * expert_gate
                - 0.08 * anti_interaction
            )
            discuss = logits["discuss_type"].clone()
            discuss[:, self.data_idx] = discuss[:, self.data_idx] + scale * route_score - 0.30 * scale * negative_gate
            suppress = suppress_scale * positive_gate * (0.65 + 0.35 * torch.relu(layout))
            for idx in range(discuss.shape[1]):
                if idx != self.data_idx:
                    discuss[:, idx] = discuss[:, idx] - suppress
            hard_competitor_suppress = 0.55 * suppress_scale * torch.relu(expert_gate - 0.45) * (0.70 + 0.30 * positive_gate)
            discuss[:, self.debate_idx] = discuss[:, self.debate_idx] - hard_competitor_suppress
            discuss[:, self.socratic_idx] = discuss[:, self.socratic_idx] - 0.70 * hard_competitor_suppress
            logits["discuss_type"] = discuss
            logits["data_router_score"] = route_score
        if self.socratic_cap_scale is not None and "discuss_type" in logits:
            like = logits["discuss_type"]
            p_round = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", like, detach_aux=True)
            p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
            p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
            p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
            reasoning_signal = 0.35 * p_ques + 0.30 * p_answer + 0.20 * p_discuss
            non_socratic_talk = torch.relu(0.28 * p_discuss + 0.18 * p_answer - 0.14 * p_ques)
            weak_reasoning = torch.relu(0.42 - reasoning_signal)
            cap = (0.28 + 0.72 * p_round) * (0.24 + non_socratic_talk + 0.85 * weak_reasoning)
            scale = self.socratic_cap_scale.to(device=like.device, dtype=like.dtype)
            discuss = like.clone()
            discuss[:, self.socratic_idx] = discuss[:, self.socratic_idx] - scale * cap
            logits["discuss_type"] = discuss
        if self.behavior_evidence_head is not None and "discuss_type" in logits:
            behavior_evidence = DiscussTypeEvidenceBuilder.data_evidence(logits, logits["discuss_type"], detach_aux=True)
            behavior_logits, behavior_delta = self.behavior_evidence_head(behavior_evidence)
            logits["behavior_evidence_discuss"] = behavior_logits
            logits["discuss_type"] = logits["discuss_type"] + behavior_delta
        if self.pair_override_head is not None and self.disentangled_evidence_adapter is None and "discuss_type" in logits:
            pair_logits, pair_delta = self.pair_override_head(feat)
            logits["guide_debate_override"] = pair_logits
            discuss = logits["discuss_type"].clone()
            pair_center = 0.5 * (discuss[:, self.guide_idx] + discuss[:, self.debate_idx])
            discuss[:, self.guide_idx] = pair_center + pair_delta[:, 0]
            discuss[:, self.debate_idx] = pair_center + pair_delta[:, 1]
            logits["discuss_type"] = discuss
        if self.semantic_pair_head is not None and self.disentangled_evidence_adapter is None and "discuss_type" in logits and self.semantic_tasks:
            semantic_list = build_semantic_vectors(
                logits,
                task_names=self.semantic_tasks,
                mode=self.semantic_mode,
                detach_aux=True,
            )
            semantic_vec = torch.cat(semantic_list, dim=1)
            pair_logits, pair_delta = self.semantic_pair_head(feat, semantic_vec)
            logits["guide_debate_semantic"] = pair_logits
            discuss = logits["discuss_type"].clone()
            pair_center = 0.5 * (discuss[:, self.guide_idx] + discuss[:, self.debate_idx])
            discuss[:, self.guide_idx] = pair_center + pair_delta[:, 0]
            discuss[:, self.debate_idx] = pair_center + pair_delta[:, 1]
            logits["discuss_type"] = discuss
        if self.debate_aux_guard_scale is not None and "discuss_type" in logits:
            like = logits["discuss_type"]
            p_oppo = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", like, detach_aux=True)
            p_plat = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "plat", like, detach_aux=True)
            p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
            p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
            p_mate = DiscussTypeEvidenceBuilder.prob(logits, "view", VIEW_LABELS, "mate", like, detach_aux=True)
            debate_evidence = 0.34 * p_oppo + 0.24 * p_discuss + 0.18 * p_answer + 0.16 * p_plat + 0.08 * p_mate
            deficit = torch.relu(0.55 - debate_evidence)
            scale = self.debate_aux_guard_scale.to(device=like.device, dtype=like.dtype)
            discuss = like.clone()
            discuss[:, self.debate_idx] = discuss[:, self.debate_idx] - 0.68 * scale * deficit
            logits["discuss_type"] = discuss
        if "discuss_type" in logits:
            like = logits["discuss_type"]
            discuss = like.clone()
            if self.data_router_scale is not None:
                data_semantic = DiscussTypeEvidenceBuilder.data_evidence(logits, like, detach_aux=True)
                p_com = data_semantic[:, 0]
                layout = data_semantic[:, 9]
                teacher_behavior = data_semantic[:, 10]
                student_behavior = data_semantic[:, 11]
                anti_interaction = data_semantic[:, 13]
                if "scene_desk" in logits and "scene_desk_com" in SCENE_DESK_LABELS:
                    desk_logits = logits["scene_desk"].detach()
                    com_idx = SCENE_DESK_LABELS.index("scene_desk_com")
                    other_desk = torch.cat([desk_logits[:, :com_idx], desk_logits[:, com_idx + 1:]], dim=1)
                    com_margin_gate = torch.sigmoid(desk_logits[:, com_idx] - other_desk.max(dim=1).values)
                else:
                    com_margin_gate = p_com
                if "data_specific" in logits:
                    expert_gate = torch.sigmoid(logits["data_specific"].detach()).to(device=like.device, dtype=like.dtype)
                else:
                    expert_gate = p_com.new_zeros(p_com.shape)
                threshold = self.data_router_threshold.to(device=like.device, dtype=like.dtype)
                scale = self.data_router_scale.to(device=like.device, dtype=like.dtype)
                suppress_scale = self.data_router_suppress_scale.to(device=like.device, dtype=like.dtype)
                margin = self.data_router_margin.to(device=like.device, dtype=like.dtype)
                com_evidence = torch.maximum(torch.maximum(p_com, com_margin_gate), expert_gate)
                positive_gate = torch.relu(com_evidence - threshold)
                negative_gate = torch.relu(threshold - com_evidence)
                route_score = positive_gate * (
                    1.45
                    + 0.52 * teacher_behavior
                    + 0.42 * student_behavior
                    + 0.32 * torch.relu(layout)
                    + 0.34 * com_margin_gate
                    + 0.42 * expert_gate
                    - 0.06 * anti_interaction
                )
                discuss[:, self.data_idx] = discuss[:, self.data_idx] + 0.88 * scale * route_score - 0.18 * scale * negative_gate
                suppress = 0.65 * suppress_scale * positive_gate * (0.60 + 0.40 * torch.relu(layout))
                for idx in range(discuss.shape[1]):
                    if idx != self.data_idx:
                        discuss[:, idx] = discuss[:, idx] - suppress
                hard_competitor_suppress = 0.70 * suppress_scale * torch.relu(expert_gate - 0.45) * (0.70 + 0.30 * positive_gate)
                discuss[:, self.debate_idx] = discuss[:, self.debate_idx] - hard_competitor_suppress
                discuss[:, self.socratic_idx] = discuss[:, self.socratic_idx] - 0.70 * hard_competitor_suppress
                if float(self.data_router_margin.item()) > 0:
                    other_idx = [i for i in range(discuss.shape[1]) if i != self.data_idx]
                    other_max = discuss[:, other_idx].max(dim=1).values
                    deficit = torch.relu(margin - (discuss[:, self.data_idx] - other_max))
                    discuss[:, self.data_idx] = discuss[:, self.data_idx] + 0.75 * scale * positive_gate * deficit
                    soft_data_support = torch.relu(
                        0.34 * p_com
                        + 0.24 * com_margin_gate
                        + 0.30 * expert_gate
                        + 0.18 * teacher_behavior
                        + 0.14 * student_behavior
                        + 0.12 * torch.relu(layout)
                        - 0.10 * anti_interaction
                        - 0.14
                    )
                    soft_deficit = torch.relu(0.65 * margin - (discuss[:, self.data_idx] - other_max))
                    discuss[:, self.data_idx] = discuss[:, self.data_idx] + 0.42 * scale * soft_data_support * soft_deficit
                    soft_suppress = 0.18 * suppress_scale * soft_data_support
                    discuss[:, self.debate_idx] = discuss[:, self.debate_idx] - soft_suppress
                    discuss[:, self.question_idx] = discuss[:, self.question_idx] - 0.85 * soft_suppress
                    discuss[:, self.guide_idx] = discuss[:, self.guide_idx] - 0.85 * soft_suppress
                logits["data_router_final_score"] = route_score
            if self.question_router_scale is not None:
                p_plat = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "plat", like, detach_aux=True)
                p_group = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", like, detach_aux=True)
                p_round = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", like, detach_aux=True)
                p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
                p_patrol = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", like, detach_aux=True)
                p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
                p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
                q_score = torch.relu((0.18 + 0.82 * p_plat) * (
                    0.36 * p_ques + 0.30 * p_answer + 0.26 * p_discuss + 0.10 * p_patrol + 0.12 * p_group
                ) - 0.10 * p_round)
                scale = self.question_router_scale.to(device=like.device, dtype=like.dtype)
                discuss[:, self.question_idx] = discuss[:, self.question_idx] + 0.90 * scale * q_score
                discuss[:, self.guide_idx] = discuss[:, self.guide_idx] - 0.42 * scale * q_score
                discuss[:, self.socratic_idx] = discuss[:, self.socratic_idx] - 0.22 * scale * q_score
                logits["question_router_final_score"] = q_score
            if self.guide_cap_scale is not None:
                p_plat = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "plat", like, detach_aux=True)
                p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
                p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
                p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
                question_like = p_plat * (0.40 * p_ques + 0.32 * p_answer + 0.26 * p_discuss + 0.10)
                scale = self.guide_cap_scale.to(device=like.device, dtype=like.dtype)
                discuss[:, self.guide_idx] = discuss[:, self.guide_idx] - 0.85 * scale * question_like
            if self.socratic_cap_scale is not None:
                p_round = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", like, detach_aux=True)
                p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
                p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
                p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
                p_group = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", like, detach_aux=True)
                p_oppo = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", like, detach_aux=True)
                p_com = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_com", like, detach_aux=True)
                reasoning_signal = 0.35 * p_ques + 0.30 * p_answer + 0.20 * p_discuss
                weak_reasoning = torch.relu(0.42 - reasoning_signal)
                debate_or_data_competition = torch.relu(0.44 * p_oppo + 0.20 * p_group + 0.16 * p_com + 0.14 * p_discuss - 0.30 * p_ques)
                round_shortcut = torch.relu(p_round - reasoning_signal)
                cap = (0.24 + 0.76 * p_round) * (
                    0.26
                    + 0.45 * torch.relu(0.28 * p_discuss + 0.20 * p_answer - 0.16 * p_ques)
                    + 0.90 * weak_reasoning
                    + 0.55 * debate_or_data_competition
                    + 0.35 * round_shortcut
                )
                scale = self.socratic_cap_scale.to(device=like.device, dtype=like.dtype)
                discuss[:, self.socratic_idx] = discuss[:, self.socratic_idx] - 0.92 * scale * cap
            logits["discuss_type"] = discuss
        if self.evidence_competition_scale is not None and "discuss_type" in logits:
            like = logits["discuss_type"]
            evidence_scores = DiscussTypeEvidenceBuilder.competition_scores(logits, like, detach_aux=True)
            evidence_delta = evidence_scores - evidence_scores.mean(dim=1, keepdim=True)
            scale = self.evidence_competition_scale.to(device=like.device, dtype=like.dtype)
            logits["discuss_type"] = like + scale * evidence_delta
            logits["evidence_competition_scores"] = evidence_scores
        if "discuss_type" in logits and (
            self.debate_temper_scale is not None
            or self.question_temper_scale is not None
            or self.socratic_recall_boost_scale is not None
        ):
            like = logits["discuss_type"]
            p_group = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_group", like, detach_aux=True)
            p_oppo = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_oppo", like, detach_aux=True)
            p_round = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_round", like, detach_aux=True)
            p_com = DiscussTypeEvidenceBuilder.prob(logits, "scene_desk", SCENE_DESK_LABELS, "scene_desk_com", like, detach_aux=True)
            p_plat = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "plat", like, detach_aux=True)
            p_under = DiscussTypeEvidenceBuilder.prob(logits, "location", LOCATION_LABELS, "under", like, detach_aux=True)
            p_ques = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_ques", like, detach_aux=True)
            p_guide_act = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_guide", like, detach_aux=True)
            p_patrol = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_patrol", like, detach_aux=True)
            p_exp = DiscussTypeEvidenceBuilder.prob(logits, "teacher_act", TEACHER_ACT_LABELS, "teacher_act_exp", like, detach_aux=True)
            p_answer = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_answer", like, detach_aux=True)
            p_discuss = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_discuss", like, detach_aux=True)
            p_listen = DiscussTypeEvidenceBuilder.prob(logits, "stu_act", STU_ACT_LABELS, "stu_act_listen", like, detach_aux=True)
            p_mate = DiscussTypeEvidenceBuilder.prob(logits, "view", VIEW_LABELS, "mate", like, detach_aux=True)
            discuss = like.clone()
            if self.debate_temper_scale is not None:
                data_semantic = DiscussTypeEvidenceBuilder.data_evidence(logits, like, detach_aux=True)
                data_layout = data_semantic[:, 9]
                data_teacher = data_semantic[:, 10]
                data_student = data_semantic[:, 11]
                data_anti = data_semantic[:, 13]
                if "data_specific" in logits:
                    data_expert = torch.sigmoid(logits["data_specific"].detach()).to(device=like.device, dtype=like.dtype)
                else:
                    data_expert = like.new_zeros(like.shape[0])
                debate_core = 0.42 * p_oppo + 0.20 * p_mate + 0.16 * p_discuss + 0.10 * p_answer + 0.06 * p_listen
                debate_shortcut = 0.46 * p_round + 0.24 * p_group + 0.18 * p_com + 0.08 * p_plat + 0.05 * p_under
                debate_specificity = 0.38 * p_oppo + 0.24 * p_mate + 0.20 * p_discuss + 0.12 * p_answer + 0.06 * p_listen
                weak_debate_specificity = torch.relu(0.58 - debate_specificity) * torch.relu(0.62 - p_oppo - 0.30 * p_mate)
                data_conflict = torch.relu(
                    0.44 * p_com
                    + 0.25 * torch.relu(data_layout)
                    + 0.22 * data_teacher
                    + 0.18 * data_student
                    + 0.54 * data_expert
                    - 0.22 * data_anti
                    - debate_core
                    + 0.02
                )
                guide_core = 0.34 * p_group + 0.24 * p_under + 0.18 * p_guide_act + 0.14 * p_patrol + 0.06 * p_exp
                data_guide_conflict = torch.relu(
                    0.40 * p_com
                    + 0.18 * torch.relu(data_layout)
                    + 0.18 * data_teacher
                    + 0.12 * data_student
                    + 0.44 * data_expert
                    - guide_core
                    - 0.03
                )
                debate_overreach = torch.relu(debate_shortcut - debate_core + 0.18) + 0.42 * weak_debate_specificity
                scale = self.debate_temper_scale.to(device=like.device, dtype=like.dtype)
                discuss[:, self.debate_idx] = discuss[:, self.debate_idx] - scale * debate_overreach
                conflict_scale = torch.clamp(scale, max=2.25)
                discuss[:, self.debate_idx] = discuss[:, self.debate_idx] - 0.95 * conflict_scale * data_conflict
                discuss[:, self.data_idx] = discuss[:, self.data_idx] + 0.55 * conflict_scale * data_conflict
                discuss[:, self.guide_idx] = discuss[:, self.guide_idx] - 0.55 * conflict_scale * data_guide_conflict
                discuss[:, self.data_idx] = discuss[:, self.data_idx] + 0.35 * conflict_scale * data_guide_conflict
            if self.question_temper_scale is not None:
                data_semantic_q = DiscussTypeEvidenceBuilder.data_evidence(logits, like, detach_aux=True)
                data_layout_q = data_semantic_q[:, 9]
                data_teacher_q = data_semantic_q[:, 10]
                data_student_q = data_semantic_q[:, 11]
                if "data_specific" in logits:
                    data_expert_q = torch.sigmoid(logits["data_specific"].detach()).to(device=like.device, dtype=like.dtype)
                else:
                    data_expert_q = like.new_zeros(like.shape[0])
                debate_core_q = 0.38 * p_oppo + 0.24 * p_mate + 0.18 * p_discuss + 0.12 * p_answer + 0.08 * p_listen
                question_evidence = 0.34 * p_plat + 0.24 * p_group + 0.20 * p_ques + 0.14 * p_answer + 0.10 * p_discuss
                data_or_debate_evidence = torch.maximum(
                    0.44 * p_com + 0.18 * torch.relu(data_layout_q) + 0.16 * data_teacher_q + 0.12 * data_student_q + 0.34 * data_expert_q,
                    debate_core_q,
                )
                data_question_conflict = torch.relu(
                    0.36 * p_com
                    + 0.18 * torch.relu(data_layout_q)
                    + 0.14 * data_teacher_q
                    + 0.12 * data_student_q
                    + 0.40 * data_expert_q
                    - question_evidence
                    - 0.02
                )
                question_overreach = torch.relu(
                    0.30 * p_round
                    + 0.22 * p_oppo
                    + 0.18 * p_com
                    + 0.08 * p_under
                    + 0.38 * torch.relu(data_or_debate_evidence - question_evidence)
                    + 0.45 * data_question_conflict
                    - 0.28 * p_group
                    - 0.18 * p_plat
                )
                scale = self.question_temper_scale.to(device=like.device, dtype=like.dtype)
                discuss[:, self.question_idx] = discuss[:, self.question_idx] - scale * question_overreach
            if self.socratic_recall_boost_scale is not None:
                socratic_recall = torch.relu(0.56 * p_round + 0.18 * p_ques + 0.14 * p_answer + 0.12 * p_discuss - 0.26 * p_oppo - 0.16 * p_com)
                scale = self.socratic_recall_boost_scale.to(device=like.device, dtype=like.dtype)
                discuss[:, self.socratic_idx] = discuss[:, self.socratic_idx] + scale * socratic_recall
            logits["discuss_type"] = discuss
        return logits


def build_feature_backbone(backbone: str, pretrained: bool = True, pretrained_path: str = ""):
    """构建 torchvision 视频 backbone，并去掉原分类头。"""
    backbone = str(backbone).lower()
    try:
        from torchvision.models.video import (
            mc3_18,
            MC3_18_Weights,
            mvit_v2_s,
            MViT_V2_S_Weights,
            r2plus1d_18,
            R2Plus1D_18_Weights,
            r3d_18,
            R3D_18_Weights,
            swin3d_t,
            Swin3D_T_Weights,
            s3d,
            S3D_Weights,
        )
    except Exception as e:
        raise RuntimeError("需要 torchvision 支持 video models") from e

    def safe_build(builder, weights):
        if pretrained_path:
            return builder(weights=None)
        if not pretrained:
            return builder(weights=None)
        try:
            return builder(weights=weights)
        except Exception as e:
            print(f"[pretrained:warning] failed to load/download {backbone} pretrained weights: {e}; fallback to random init. Use --no_pretrained to silence this.")
            return builder(weights=None)

    def load_local_pretrained(model: nn.Module) -> nn.Module:
        if not pretrained_path:
            return model
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported pretrained checkpoint format: {pretrained_path}")
        cleaned = {}
        for k, v in state.items():
            kk = str(k)
            for prefix in ("module.", "model.", "backbone."):
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
            cleaned[kk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[pretrained:local] loaded {pretrained_path}; missing={len(missing)} unexpected={len(unexpected)}")
        return model

    if backbone in ("i3d", "s3d"):
        base = safe_build(s3d, S3D_Weights.KINETICS400_V1)
        base = load_local_pretrained(base)
        feat_dim = base.classifier[1].in_channels
        # Torchvision S3D uses a fixed 7x7 spatial AvgPool3d, which fails for
        # our 112x112 classroom clips after backbone downsampling. Adaptive
        # pooling keeps the backbone comparison on the same input resolution.
        base.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        base.classifier = nn.Identity()
        return base, feat_dim
    if backbone == "r3d_18":
        base = safe_build(r3d_18, R3D_18_Weights.KINETICS400_V1)
        base = load_local_pretrained(base)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        return base, feat_dim
    if backbone == "mc3_18":
        base = safe_build(mc3_18, MC3_18_Weights.KINETICS400_V1)
        base = load_local_pretrained(base)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        return base, feat_dim
    if backbone == "r2plus1d_18":
        base = safe_build(r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1)
        base = load_local_pretrained(base)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        return base, feat_dim
    if backbone == "swin3d_t":
        base = safe_build(swin3d_t, Swin3D_T_Weights.KINETICS400_V1)
        base = load_local_pretrained(base)
        feat_dim = base.head.in_features
        base.head = nn.Identity()
        return base, feat_dim
    if backbone in ("mvit_v2_s", "timesformer"):
        base = safe_build(mvit_v2_s, MViT_V2_S_Weights.KINETICS400_V1)
        base = load_local_pretrained(base)
        head = base.head
        feat_dim = head.in_features if hasattr(head, "in_features") else head[-1].in_features
        base.head = nn.Identity()
        return base, feat_dim
    if backbone == "slowfast":
        raise NotImplementedError("SlowFast 不在 torchvision 标准 video API 中；请使用 pytorchvideo 或 MMAction2 独立 baseline。")
    raise ValueError(f"unsupported backbone: {backbone}")


def build_experimental_video_model(
    num_classes_per_task: Dict[str, int],
    backbone: str = "swin3d_t",
    pretrained: bool = True,
    pretrained_path: str = "",
    fusion: str = "none",
    semantic_mode: str = "prob",
    detach_aux: bool = False,
    backbone_adapter: str = "none",
    adapter_reduction: int = 4,
    adapter_scale: float = 0.1,
    adapter_dropout: float = 0.0,
    feature_adapter: str = "none",
    pair_balance_head: bool = False,
    pair_balance_scale: float = 0.05,
    guide_specific_head: bool = False,
    guide_specific_scale: float = 0.05,
    data_specific_head: bool = False,
    data_specific_scale: float = 0.05,
    data_evidence_boost_scale: float = 0.0,
    data_router_scale: float = 0.0,
    data_router_threshold: float = 0.45,
    data_router_suppress_scale: float = 0.0,
    data_router_margin: float = 0.0,
    question_router_scale: float = 0.0,
    guide_cap_scale: float = 0.0,
    socratic_cap_scale: float = 0.0,
    guide_location_boost_scale: float = 0.0,
    debate_aux_guard_scale: float = 0.0,
    debate_temper_scale: float = 0.0,
    question_temper_scale: float = 0.0,
    socratic_recall_boost_scale: float = 0.0,
    evidence_competition_router: bool = False,
    evidence_competition_scale: float = 0.0,
    behavior_evidence_head: bool = False,
    behavior_evidence_scale: float = 0.25,
    pair_override_head: bool = False,
    pair_override_scale: float = 0.25,
    semantic_pair_head: bool = False,
    semantic_pair_scale: float = 0.25,
    disentangled_evidence_adapter: bool = False,
    disentangled_evidence_scale: float = 0.8,
    disentangled_evidence_detach_aux: bool = True,
    pedagogical_template_adapter: bool = False,
    pedagogical_template_scale: float = 0.6,
    pedagogical_template_detach_aux: bool = True,
    scene_desk_constraint_adapter: bool = False,
    scene_desk_constraint_scale: float = 0.8,
    scene_desk_constraint_detach_aux: bool = True,
    pedagogical_prior_adapter: bool = False,
    pedagogical_prior_scale: float = 0.18,
    pedagogical_prior_max_delta: float = 2.0,
    pedagogical_prior_detach_aux: bool = True,
) -> ExperimentalVideoMultiTaskModel:
    return ExperimentalVideoMultiTaskModel(
        num_classes_per_task=num_classes_per_task,
        backbone=backbone,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        dropout=0.3,
        fusion=fusion,
        semantic_mode=semantic_mode,
        detach_aux=detach_aux,
        backbone_adapter=backbone_adapter,
        adapter_reduction=adapter_reduction,
        adapter_scale=adapter_scale,
        adapter_dropout=adapter_dropout,
        feature_adapter=feature_adapter,
        pair_balance_head=pair_balance_head,
        pair_balance_scale=pair_balance_scale,
        guide_specific_head=guide_specific_head,
        guide_specific_scale=guide_specific_scale,
        data_specific_head=data_specific_head,
        data_specific_scale=data_specific_scale,
        data_evidence_boost_scale=data_evidence_boost_scale,
        data_router_scale=data_router_scale,
        data_router_threshold=data_router_threshold,
        data_router_suppress_scale=data_router_suppress_scale,
        data_router_margin=data_router_margin,
        question_router_scale=question_router_scale,
        guide_cap_scale=guide_cap_scale,
        socratic_cap_scale=socratic_cap_scale,
        guide_location_boost_scale=guide_location_boost_scale,
        debate_aux_guard_scale=debate_aux_guard_scale,
        debate_temper_scale=debate_temper_scale,
        question_temper_scale=question_temper_scale,
        socratic_recall_boost_scale=socratic_recall_boost_scale,
        evidence_competition_router=evidence_competition_router,
        evidence_competition_scale=evidence_competition_scale,
        behavior_evidence_head=behavior_evidence_head,
        behavior_evidence_scale=behavior_evidence_scale,
        pair_override_head=pair_override_head,
        pair_override_scale=pair_override_scale,
        semantic_pair_head=semantic_pair_head,
        semantic_pair_scale=semantic_pair_scale,
        disentangled_evidence_adapter=disentangled_evidence_adapter,
        disentangled_evidence_scale=disentangled_evidence_scale,
        disentangled_evidence_detach_aux=disentangled_evidence_detach_aux,
        pedagogical_template_adapter=pedagogical_template_adapter,
        pedagogical_template_scale=pedagogical_template_scale,
        pedagogical_template_detach_aux=pedagogical_template_detach_aux,
        scene_desk_constraint_adapter=scene_desk_constraint_adapter,
        scene_desk_constraint_scale=scene_desk_constraint_scale,
        scene_desk_constraint_detach_aux=scene_desk_constraint_detach_aux,
        pedagogical_prior_adapter=pedagogical_prior_adapter,
        pedagogical_prior_scale=pedagogical_prior_scale,
        pedagogical_prior_max_delta=pedagogical_prior_max_delta,
        pedagogical_prior_detach_aux=pedagogical_prior_detach_aux,
    )
