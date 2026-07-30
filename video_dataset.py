# -*- coding: utf-8 -*-
"""视频 clip 级 Dataset：从帧序列构造 (C, T, H, W) 输入。

面向“课堂行为识别（探讨式教学等）”的直接可跑版本：
- 使用 CVAT for video 1.1 导出的 frames + annotations.xml
- 用滑窗把连续帧组成 clip（长度 T、步长 stride、采样率 sample_rate）
- 标签默认取 clip 中心帧（center frame）的任务标签；也可改为多数投票
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_ROOT, ANNOTATION_FILE, FRAMES_DIR, TASKS
from cvat_parser import parse_cvat_video_xml, get_task_indices_and_masks


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_rgb(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _normalize(img: np.ndarray) -> np.ndarray:
    # img: HWC, uint8 or float
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


@dataclass(frozen=True)
class ClipIndex:
    start: int  # index in frame_rows
    center: int
    end: int  # exclusive


class ClassroomClipDataset(Dataset):
    """从连续帧构造 clip，用于 3D CNN 等视频模型。"""

    def __init__(
        self,
        data_root: str,
        clip_len: int = 16,
        stride: int = 8,
        sample_rate: int = 1,
        image_size: int = 112,
        frame_rows: Optional[List[dict]] = None,
    ):
        self.data_root = Path(data_root or DATA_ROOT)
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.sample_rate = int(sample_rate)
        self.image_size = int(image_size)
        self.task_names = list(TASKS.keys())
        self.num_classes = {t: len(TASKS[t][0]) for t in self.task_names}

        if frame_rows is None:
            xml_path = self.data_root / ANNOTATION_FILE
            if not xml_path.exists():
                xml_path = self.data_root / "annotations" / ANNOTATION_FILE
            if not xml_path.exists():
                raise FileNotFoundError(f"未找到 {ANNOTATION_FILE} 于 {self.data_root} 或 {self.data_root / 'annotations'}")

            frames_dir_abs = self.data_root / FRAMES_DIR
            if not frames_dir_abs.exists():
                # 兼容 images/
                if (self.data_root / "images").exists():
                    frames_dir_abs = self.data_root / "images"

            frames = parse_cvat_video_xml(str(xml_path), frames_dir=str(frames_dir_abs))
            rows = get_task_indices_and_masks(frames, str(self.data_root))
        else:
            rows = frame_rows

        # resolve image paths and sort by frame_id if possible
        resolved = []
        for r in rows:
            img_path = r["image_path"]
            p = Path(img_path)
            if not p.is_absolute():
                p = self.data_root / img_path
            if not p.exists():
                # try frames/images + basename
                for sub in (FRAMES_DIR, "images", "frames"):
                    cand = self.data_root / sub / Path(img_path).name
                    if cand.exists():
                        p = cand
                        break
            rr = dict(r)
            rr["_resolved_path"] = str(p)
            # frame_id may be str for <image id="...">
            fid = rr.get("frame_id")
            try:
                rr["_frame_int"] = int(fid)
            except Exception:
                rr["_frame_int"] = None
            resolved.append(rr)

        if all(r["_frame_int"] is not None for r in resolved):
            resolved.sort(key=lambda x: x["_frame_int"])
        self.frame_rows = resolved

        self.clips = self._build_clips()

    def _build_clips(self) -> List[ClipIndex]:
        step = self.sample_rate
        T = self.clip_len
        span = (T - 1) * step + 1
        n = len(self.frame_rows)
        clips: List[ClipIndex] = []
        # sliding window over indices
        for start in range(0, max(n - span + 1, 0), self.stride):
            end = start + span
            center = start + (span // 2)
            if end <= n:
                clips.append(ClipIndex(start=start, center=center, end=end))
        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        c = self.clips[idx]
        # sample indices within [start,end) using sample_rate
        indices = list(range(c.start, c.end, self.sample_rate))
        assert len(indices) == self.clip_len

        frames = []
        for i in indices:
            path = self.frame_rows[i]["_resolved_path"]
            img = _read_rgb(path)
            img = _resize(img, self.image_size)
            img = _normalize(img)  # HWC float32
            frames.append(img)

        # stack to T,H,W,C then to C,T,H,W
        arr = np.stack(frames, axis=0)  # T,H,W,C
        arr = np.transpose(arr, (3, 0, 1, 2))  # C,T,H,W
        video = torch.from_numpy(arr).float()

        # label = center frame
        center_row = self.frame_rows[c.center]
        out: Dict[str, torch.Tensor] = {"video": video}
        for t in self.task_names:
            out[f"{t}_idx"] = torch.tensor(center_row.get(f"{t}_idx", -1), dtype=torch.long)
            out[f"{t}_valid"] = torch.tensor(center_row.get(f"{t}_valid", False), dtype=torch.bool)
        return out


def default_collate_video(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {}
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out


def build_video_dataloaders(
    data_root: str,
    batch_size: int = 4,
    clip_len: int = 16,
    stride: int = 8,
    sample_rate: int = 1,
    image_size: int = 112,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, int]]:
    import random

    full = ClassroomClipDataset(
        data_root=data_root,
        clip_len=clip_len,
        stride=stride,
        sample_rate=sample_rate,
        image_size=image_size,
    )
    n = len(full)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(n * train_ratio)

    # split clips
    train_clips = [full.clips[i] for i in idxs[:n_train]]
    val_clips = [full.clips[i] for i in idxs[n_train:]]

    train_ds = _SubsetClipDataset(full, train_clips)
    val_ds = _SubsetClipDataset(full, val_clips)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=default_collate_video,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=default_collate_video,
    )
    return train_loader, val_loader, full.task_names, full.num_classes


class _SubsetClipDataset(Dataset):
    """不复制 frame_rows，只替换 clips 的轻量子集。"""

    def __init__(self, base: ClassroomClipDataset, clips: List[ClipIndex]):
        self.base = base
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 临时替换 base.clips 的索引方式
        c = self.clips[idx]
        # 复用 base 的逻辑：直接手写一份避免修改 base 状态
        indices = list(range(c.start, c.end, self.base.sample_rate))
        frames = []
        for i in indices:
            path = self.base.frame_rows[i]["_resolved_path"]
            img = _read_rgb(path)
            img = _resize(img, self.base.image_size)
            img = _normalize(img)
            frames.append(img)
        arr = np.stack(frames, axis=0)
        arr = np.transpose(arr, (3, 0, 1, 2))
        video = torch.from_numpy(arr).float()

        center_row = self.base.frame_rows[c.center]
        out: Dict[str, torch.Tensor] = {"video": video}
        for t in self.base.task_names:
            out[f"{t}_idx"] = torch.tensor(center_row.get(f"{t}_idx", -1), dtype=torch.long)
            out[f"{t}_valid"] = torch.tensor(center_row.get(f"{t}_valid", False), dtype=torch.bool)
        return out

