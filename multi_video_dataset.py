# -*- coding: utf-8 -*-
"""多视频 clip 数据集 - 支持过采样（guide 视频 2 倍）"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DISCUSS_TYPE_BY_VIDEO_1BASED, DISCUSS_TYPE_EXTRA_CORRECT_BY_VIDEO_1BASED, DISCUSS_TYPE_TO_IDX, TASKS
from cvat_parser import get_task_indices_and_masks, parse_cvat_video_xml

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_rgb(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _normalize(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@dataclass(frozen=True)
class ClipRef:
    video_id: int
    start: int
    center: int
    end: int


class MultiVideoClipDataset(Dataset):
    """将多个视频目录合并为一个 clip 数据集，支持过采样"""

    def __init__(
        self,
        root_dir: str,
        clip_len: int = 16,
        stride: int = 8,
        sample_rate: int = 1,
        image_size: int = 112,
        label_aggregation: str = "center",
        min_valid_frames_by_task: Optional[Dict[str, int]] = None,
        video_dirs: Optional[List[str]] = None,
        oversample_factors: Optional[Dict[int, int]] = None,
        guide_location_rule_videos: Optional[List[int]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.sample_rate = int(sample_rate)
        self.image_size = int(image_size)
        self.label_aggregation = str(label_aggregation).lower()
        if self.label_aggregation not in ("center", "majority"):
            raise ValueError(f"label_aggregation must be 'center' or 'majority', got: {label_aggregation}")

        self.task_names = list(TASKS.keys())
        self.num_classes = {t: len(TASKS[t][0]) for t in self.task_names}

        if min_valid_frames_by_task is None:
            min_valid_frames_by_task = {"teacher_act": 2, "stu_act": 2, "view": 2}
        self.min_valid_frames_by_task = {str(k): int(v) for k, v in min_valid_frames_by_task.items()}
        self.guide_location_rule_videos = set(int(x) for x in (guide_location_rule_videos or []))
        env_rule = os.environ.get("CLASSROOM_GUIDE_LOCATION_VIDEOS", "").strip()
        if env_rule:
            for part in env_rule.replace(";", ",").split(","):
                part = part.strip()
                if part:
                    self.guide_location_rule_videos.add(int(part))

        self.skip_clips = mp.Value("i", 0)
        self.skip_frames = mp.Value("i", 0)

        self.video_dirs = self._discover_video_dirs(video_dirs)
        self.videos = self._load_videos()
        self.oversample_factors = oversample_factors or {}
        self.clips = self._build_all_clips()
        self.video_discuss_type_idx = self._build_video_discuss_type_idx()
        self.video_discuss_type_extra_correct = self._build_video_discuss_type_extra_correct()

    def _discover_video_dirs(self, video_dirs: Optional[List[str]]) -> List[Path]:
        if video_dirs is not None:
            return [self.root_dir / d for d in video_dirs]
        candidates = []
        for p in self.root_dir.iterdir():
            if not p.is_dir() or p.name.startswith("."):
                continue
            xml_ok = (p / "annotations.xml").exists() or (p / "annotations" / "annotations.xml").exists()
            img_ok = (p / "images").exists() or (p / "frames").exists()
            if xml_ok and img_ok:
                candidates.append(p)
        candidates.sort()
        if not candidates:
            raise FileNotFoundError(f"未找到有效视频子目录: {self.root_dir}")
        return candidates

    def _load_videos(self) -> List[dict]:
        videos = []
        for vdir in self.video_dirs:
            xml_path = vdir / "annotations.xml"
            if not xml_path.exists():
                xml_path = vdir / "annotations" / "annotations.xml"
            if not xml_path.exists():
                raise FileNotFoundError(f"视频目录缺少 annotations.xml: {vdir}")
            images_dir = vdir / "images"
            if not images_dir.exists():
                images_dir = vdir / "frames"
            if not images_dir.exists():
                raise FileNotFoundError(f"视频目录缺少 images/ 或 frames/: {vdir}")
            files_by_stem = {fp.stem.lower(): fp for fp in images_dir.iterdir() if fp.is_file()}
            frames = parse_cvat_video_xml(str(xml_path), frames_dir=str(images_dir))
            rows = get_task_indices_and_masks(frames, str(vdir))
            resolved = []
            for r in rows:
                rr = dict(r)
                rr["_stem"] = Path(r["image_path"]).stem.lower()
                try:
                    rr["_frame_int"] = int(rr.get("frame_id"))
                except Exception:
                    rr["_frame_int"] = None
                resolved.append(rr)
            if all(r["_frame_int"] is not None for r in resolved):
                resolved.sort(key=lambda x: x["_frame_int"])
            videos.append({
                "dir": vdir,
                "images_dir": images_dir,
                "frame_rows": resolved,
                "files_by_stem": files_by_stem,
            })
        return videos

    def _build_video_discuss_type_idx(self) -> Dict[int, int]:
        out = {}
        for video_id, _ in enumerate(self.videos):
            label = DISCUSS_TYPE_BY_VIDEO_1BASED.get(video_id + 1)
            out[video_id] = DISCUSS_TYPE_TO_IDX.get(label, -1) if label is not None else -1
        return out

    def _build_video_discuss_type_extra_correct(self) -> Dict[int, List[int]]:
        out = {}
        for video_id, _ in enumerate(self.videos):
            labels = DISCUSS_TYPE_EXTRA_CORRECT_BY_VIDEO_1BASED.get(video_id + 1, [])
            out[video_id] = [DISCUSS_TYPE_TO_IDX[x] for x in labels if x in DISCUSS_TYPE_TO_IDX]
        return out

    def _use_guide_location_rule(self, video_id: int) -> bool:
        one_based = int(video_id) + 1
        if self.guide_location_rule_videos:
            return one_based in self.guide_location_rule_videos
        guide_idx = DISCUSS_TYPE_TO_IDX.get("guide_discuss", -999)
        question_idx = DISCUSS_TYPE_TO_IDX.get("question_discuss", -999)
        primary = int(self.video_discuss_type_idx.get(int(video_id), -1)) if hasattr(self, "video_discuss_type_idx") else -1
        extras = self.video_discuss_type_extra_correct.get(int(video_id), []) if hasattr(self, "video_discuss_type_extra_correct") else []
        auto_mixed = primary == question_idx and guide_idx in extras
        return one_based in self.guide_location_rule_videos or auto_mixed

    def _resolve_discuss_type_for_clip(self, video_id: int, out: dict) -> tuple[int, bool]:
        discuss_idx = int(self.video_discuss_type_idx.get(video_id, -1))
        if not self._use_guide_location_rule(video_id):
            return discuss_idx, False
        if "location_valid" not in out or "location_idx" not in out or not bool(out["location_valid"].item()):
            return discuss_idx, False
        loc_idx = int(out["location_idx"].item())
        location_to_idx = TASKS["location"][1]
        if loc_idx == location_to_idx.get("under", -999):
            return int(DISCUSS_TYPE_TO_IDX["guide_discuss"]), True
        if loc_idx == location_to_idx.get("plat", -999):
            return int(DISCUSS_TYPE_TO_IDX["question_discuss"]), True
        return discuss_idx, False

    def _set_discuss_type_fields(self, out: dict, video_id: int, scene_method_idx: int, scene_method_valid: bool) -> dict:
        discuss_idx, used_rule = self._resolve_discuss_type_for_clip(video_id, out)
        out["discuss_type_idx"] = torch.tensor(discuss_idx, dtype=torch.long)
        out["discuss_type_valid"] = torch.tensor(scene_method_valid and scene_method_idx == 0 and discuss_idx >= 0, dtype=torch.bool)
        multi = torch.zeros(int(self.num_classes["discuss_type"]), dtype=torch.bool)
        if 0 <= discuss_idx < int(self.num_classes["discuss_type"]):
            multi[discuss_idx] = True
        if not used_rule:
            for extra_idx in self.video_discuss_type_extra_correct.get(video_id, []):
                if 0 <= int(extra_idx) < int(self.num_classes["discuss_type"]):
                    multi[int(extra_idx)] = True
        out["discuss_type_multi_hot"] = multi
        out["discuss_type_location_rule"] = torch.tensor(bool(used_rule), dtype=torch.bool)
        return out

    def _build_all_clips(self) -> List[ClipRef]:
        T = self.clip_len
        step = self.sample_rate
        span = (T - 1) * step + 1
        out = []
        for video_id, v in enumerate(self.videos):
            n = len(v["frame_rows"])
            for start in range(0, max(n - span + 1, 0), self.stride):
                end = start + span
                center = start + (span // 2)
                if end <= n:
                    clip = ClipRef(video_id=video_id, start=start, center=center, end=end)
                    factor = self.oversample_factors.get(video_id, 1)
                    for _ in range(factor):
                        out.append(clip)
        return out

    def _add_clip_label_distributions(self, out: dict, rows: list, indices: list[int]) -> dict:
        for t in self.task_names:
            ncls = int(self.num_classes[t])
            hist = np.zeros(ncls, dtype=np.float32)
            valid_count = 0
            for i in indices:
                if bool(rows[i].get(f"{t}_valid", False)):
                    tidx = int(rows[i].get(f"{t}_idx", -1))
                    if 0 <= tidx < ncls:
                        hist[tidx] += 1.0
                        valid_count += 1
            if valid_count > 0:
                hist = hist / float(valid_count)
            out[f"{t}_soft"] = torch.from_numpy(hist).float()
            out[f"{t}_frame_valid_count"] = torch.tensor(valid_count, dtype=torch.long)
        return out

    def __len__(self) -> int:
        return len(self.clips)

    def _build_label_fields(self, out: dict, c: ClipRef, rows: list, indices: list[int]) -> dict:
        if self.label_aggregation == "center":
            center_row = rows[c.center]
            for t in self.task_names:
                out[f"{t}_idx"] = torch.tensor(center_row.get(f"{t}_idx", -1), dtype=torch.long)
                out[f"{t}_valid"] = torch.tensor(center_row.get(f"{t}_valid", False), dtype=torch.bool)
                if t in ("teacher_act", "stu_act", "view"):
                    ncls = int(self.num_classes[t])
                    soft = torch.zeros(ncls, dtype=torch.float32)
                    idx_val = int(center_row.get(f"{t}_idx", -1))
                    if 0 <= idx_val < ncls:
                        soft[idx_val] = 1.0
                    out[f"{t}_soft"] = soft
            self._add_clip_label_distributions(out, rows, indices)
        else:
            for t in self.task_names:
                valid_idxs = []
                for i in indices:
                    if bool(rows[i].get(f"{t}_valid", False)):
                        tidx = int(rows[i].get(f"{t}_idx", -1))
                        if tidx >= 0:
                            valid_idxs.append(tidx)
                min_valid = self.min_valid_frames_by_task.get(t, 1)
                if len(valid_idxs) < min_valid:
                    out[f"{t}_idx"] = torch.tensor(-1, dtype=torch.long)
                    out[f"{t}_valid"] = torch.tensor(False, dtype=torch.bool)
                    if t in ("teacher_act", "stu_act", "view"):
                        out[f"{t}_soft"] = torch.zeros(int(self.num_classes[t]), dtype=torch.float32)
                else:
                    counts = np.bincount(np.array(valid_idxs, dtype=np.int64))
                    mode_idx = int(counts.argmax())
                    out[f"{t}_idx"] = torch.tensor(mode_idx, dtype=torch.long)
                    out[f"{t}_valid"] = torch.tensor(True, dtype=torch.bool)
                    if t in ("teacher_act", "stu_act", "view"):
                        ncls = int(self.num_classes[t])
                        hist = np.bincount(np.array(valid_idxs, dtype=np.int64), minlength=ncls).astype(np.float32)
                        hist = hist / max(hist.sum(), 1.0)
                        out[f"{t}_soft"] = torch.from_numpy(hist).float()
            self._add_clip_label_distributions(out, rows, indices)

        scene_method_idx = int(out.get("scene_method_idx", torch.tensor(-1)).item()) if "scene_method_idx" in out else -1
        scene_method_valid = bool(out.get("scene_method_valid", torch.tensor(False)).item()) if "scene_method_valid" in out else False
        self._set_discuss_type_fields(out, c.video_id, scene_method_idx, scene_method_valid)
        return out

    def __getitem__(self, idx: int):
        c = self.clips[idx]
        v = self.videos[c.video_id]
        rows = v["frame_rows"]
        files_by_stem = v["files_by_stem"]

        indices = list(range(c.start, c.end, self.sample_rate))
        frames = []
        try:
            for i in indices:
                stem = rows[i]["_stem"]
                fp = files_by_stem.get(stem)
                if fp is None:
                    with self.skip_frames.get_lock():
                        self.skip_frames.value += 1
                    raise FileNotFoundError(f"missing frame stem={stem} in {v['images_dir']}")
                img = _read_rgb(str(fp))
                img = _resize(img, self.image_size)
                img = _normalize(img)
                frames.append(img)
        except Exception:
            with self.skip_clips.get_lock():
                self.skip_clips.value += 1
            return None

        arr = np.stack(frames, axis=0)  # T,H,W,C
        arr = np.transpose(arr, (3, 0, 1, 2))  # C,T,H,W
        video = torch.from_numpy(arr).float()

        out = {
            "video": video,
            "video_id": torch.tensor(c.video_id, dtype=torch.long),
            "clip_idx": torch.tensor(idx, dtype=torch.long),
            "clip_start": torch.tensor(c.start, dtype=torch.long),
            "clip_center": torch.tensor(c.center, dtype=torch.long),
            "clip_end": torch.tensor(c.end, dtype=torch.long),
        }
        return self._build_label_fields(out, c, rows, indices)


def default_collate_video(batch: List[dict]) -> Dict[str, torch.Tensor]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def make_loader(
    dataset: Dataset,
    indices: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
) -> DataLoader:
    from torch.utils.data import Subset

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": default_collate_video,
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(Subset(dataset, indices), **loader_kwargs)
