# -*- coding: utf-8 -*-
"""Configuration file for label definitions, paths, and training defaults."""

import os
from pathlib import Path

# Data paths. Update these paths on the target server if needed.
# Expected CVAT export layout after extraction: data_root/annotations.xml and data_root/frames or data_root/images.
DATA_ROOT = os.environ.get("CLASSROOM_DATA_ROOT", "./data")
ANNOTATION_FILE = "annotations.xml"
FRAMES_DIR = "frames"

# Scene labels. These are frame-level annotations.
SCENE_DESK_LABELS = [
    "scene_desk_group",   # group seating
    "scene_desk_round",   # round/opposite seating
    "scene_desk_oppo",    # opposite/debate seating, aligned with scene semantics
    "scene_desk_com",     # row-style or computer-room layout
]

SCENE_METHOD_LABELS = [
    "scene_method_discuss",  # discussion-based teaching
    "scene_method_teach",    # lecture-based teaching
]

SCENE_INTE_LABELS = [
    "scene_inte_group",  # group interaction
    "scene_inte_oto",    # one-to-one interaction
]

# Teacher labels. The location task is annotated as an attribute of teacher-action shapes/tracks in CVAT.
TEACHER_ACT_LABELS = [
    "teacher_act_exp",     # explaining
    "teacher_act_ques",    # questioning
    "teacher_act_guide",   # guiding
    "teacher_act_listen",  # listening
    "teacher_act_patrol",  # patrolling
]

LOCATION_LABELS = [
    "plat",   # platform
    "under",  # off-platform / classroom floor
]

# Student labels. The view task is annotated as an attribute of student-action shapes/tracks in CVAT.
STU_ACT_LABELS = [
    "stu_act_answer",   # answering
    "stu_act_write",    # writing
    "stu_act_discuss",  # discussing
    "stu_act_listen",   # listening
]

VIEW_LABELS = [
    "mate",     # peer interaction
    "teacher",  # looking at teacher
]

DISCUSS_TYPE_LABELS = [
    "question_discuss",
    "guide_discuss",
    "debate_discuss",
    "socratic_discuss",
    "data_discuss",
]

DISCUSS_TYPE_BY_VIDEO_1BASED = {
    1: "socratic_discuss",
    2: "question_discuss",
    3: "socratic_discuss",
    4: "socratic_discuss",
    6: "data_discuss",
    7: "question_discuss",
    8: "debate_discuss",
    9: "question_discuss",
}

DISCUSS_TYPE_EXTRA_CORRECT_BY_VIDEO_1BASED = {
    7: ["guide_discuss"],
    9: ["guide_discuss"],
}


def get_label_to_idx(labels):
    """Map label names to integer indices."""
    return {name: i for i, name in enumerate(labels)}


SCENE_DESK_TO_IDX = get_label_to_idx(SCENE_DESK_LABELS)
SCENE_METHOD_TO_IDX = get_label_to_idx(SCENE_METHOD_LABELS)
SCENE_INTE_TO_IDX = get_label_to_idx(SCENE_INTE_LABELS)
TEACHER_ACT_TO_IDX = get_label_to_idx(TEACHER_ACT_LABELS)
LOCATION_TO_IDX = get_label_to_idx(LOCATION_LABELS)
STU_ACT_TO_IDX = get_label_to_idx(STU_ACT_LABELS)
VIEW_TO_IDX = get_label_to_idx(VIEW_LABELS)
DISCUSS_TYPE_TO_IDX = get_label_to_idx(DISCUSS_TYPE_LABELS)

# Aliases for historical misspellings in CVAT annotations.
ALIAS_MAP = {
    "scne_mothod_discuss": "scene_method_discuss",
    "scne_mothod_teach": "scene_method_teach",
    "scne_inte_group": "scene_inte_group",
    "scne_inte_oto": "scene_inte_oto",
    "stu_act_nswer": "stu_act_answer",
}

# Task definitions used by multi-task heads.
TASKS = {
    "scene_desk":   (SCENE_DESK_LABELS, SCENE_DESK_TO_IDX),
    "scene_method": (SCENE_METHOD_LABELS, SCENE_METHOD_TO_IDX),
    "scene_inte":   (SCENE_INTE_LABELS, SCENE_INTE_TO_IDX),
    "teacher_act":  (TEACHER_ACT_LABELS, TEACHER_ACT_TO_IDX),
    "location":     (LOCATION_LABELS, LOCATION_TO_IDX),
    "stu_act":      (STU_ACT_LABELS, STU_ACT_TO_IDX),
    "view":         (VIEW_LABELS, VIEW_TO_IDX),
    "discuss_type": (DISCUSS_TYPE_LABELS, DISCUSS_TYPE_TO_IDX),
}

# Multi-task loss weights and label smoothing.
# Behavior-related tasks are harder and more imbalanced, so their losses are moderately up-weighted.
TASK_LOSS_WEIGHTS = {
    "scene_desk": 0.6,
    "scene_method": 1.0,
    "scene_inte": 1.0,
    "teacher_act": 2.8,
    "location": 1.8,
    "stu_act": 2.8,
    "view": 1.8,
    "discuss_type": 3.0,
}

# Label smoothing coefficient used by cross-entropy losses.
LABEL_SMOOTHING = 0.05

# Tasks that use soft targets. Clip-level behavior labels may be temporally discontinuous and noisy.
# Soft target supervision uses the label distribution inside a clip to reduce hard-label noise.
SOFT_LABEL_TASKS = ["teacher_act", "stu_act", "view", "location", "scene_inte", "scene_desk"]
SOFT_LABEL_ALPHA = 0.7

# Training and evaluation defaults.
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = 16


def _default_device() -> str:
    """Prefer npu > cuda > cpu to avoid silently missing available acceleration."""
    try:
        import torch
        try:
            import torch_npu  # noqa: F401
        except Exception:
            torch_npu = None  # type: ignore
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


DEVICE = _default_device()
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
OUTPUT_DIR = Path("./outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
