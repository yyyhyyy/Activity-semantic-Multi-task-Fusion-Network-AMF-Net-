# -*- coding: utf-8 -*-
"""AFS: Adaptive Frame Selection dataset wrapper.

AFS estimates motion strength from differences between neighboring frames and
selects frames with stronger motion information from long videos or long clips.
This reduces redundant frame computation. The implementation is a lightweight
CPU sampling strategy and does not depend on hard-coded rule labels.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from multi_video_dataset import MultiVideoClipDataset, _normalize, _read_rgb, _resize


class AFSMultiVideoClipDataset(MultiVideoClipDataset):
    """Adaptive key-frame sampling dataset based on motion strength."""

    def __init__(self, *args, afs_candidates: int = 32, afs_topk: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.afs_candidates = max(int(afs_candidates), self.clip_len)
        self.afs_topk = int(afs_topk or self.clip_len)
        if self.afs_topk != self.clip_len:
            raise ValueError("The current model binds input T to clip_len, so afs_topk must equal clip_len")

    def _select_indices_by_motion(self, rows: list, files_by_stem: dict, start: int, end: int) -> List[int]:
        span_indices = np.linspace(start, end - 1, num=min(self.afs_candidates, end - start), dtype=np.int64).tolist()
        if len(span_indices) <= self.clip_len:
            return list(np.linspace(start, end - 1, num=self.clip_len, dtype=np.int64))

        grays = []
        import cv2
        for i in span_indices:
            fp = files_by_stem.get(rows[i]["_stem"])
            if fp is None:
                raise FileNotFoundError(f"missing frame stem={rows[i]['_stem']}")
            img = _read_rgb(str(fp))
            img = _resize(img, 64)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            grays.append(gray)

        scores = np.zeros(len(span_indices), dtype=np.float32)
        for i in range(1, len(grays)):
            scores[i] = float(np.mean(np.abs(grays[i] - grays[i - 1])))
        scores[0] = scores[1] if len(scores) > 1 else 0.0

        top_pos = np.argsort(scores)[-self.clip_len:]
        selected = sorted(int(span_indices[p]) for p in top_pos)
        if len(selected) < self.clip_len:
            selected = list(np.linspace(start, end - 1, num=self.clip_len, dtype=np.int64))
        return selected

    def __getitem__(self, idx: int):
        c = self.clips[idx]
        v = self.videos[c.video_id]
        rows = v["frame_rows"]
        files_by_stem = v["files_by_stem"]

        try:
            indices = self._select_indices_by_motion(rows, files_by_stem, c.start, min(c.start + self.afs_candidates * self.sample_rate, len(rows)))
            frames = []
            for i in indices:
                fp = files_by_stem.get(rows[i]["_stem"])
                if fp is None:
                    raise FileNotFoundError(f"missing frame stem={rows[i]['_stem']}")
                img = _read_rgb(str(fp))
                img = _resize(img, self.image_size)
                img = _normalize(img)
                frames.append(img)
        except Exception:
            with self.skip_clips.get_lock():
                self.skip_clips.value += 1
            return None

        arr = np.stack(frames, axis=0)
        arr = np.transpose(arr, (3, 0, 1, 2))
        video = torch.from_numpy(arr).float()

        out = {
            "video": video,
            "video_id": torch.tensor(c.video_id, dtype=torch.long),
            "clip_idx": torch.tensor(idx, dtype=torch.long),
            "clip_start": torch.tensor(c.start, dtype=torch.long),
            "clip_center": torch.tensor(c.center, dtype=torch.long),
            "clip_end": torch.tensor(c.end, dtype=torch.long),
            "afs_selected": torch.tensor(indices, dtype=torch.long),
        }
        return self._build_label_fields(out, c, rows, indices)


def build_dataset_with_sampling(sampling: str, **kwargs):
    sampling = str(sampling).lower()
    afs_candidates = int(kwargs.pop("afs_candidates", 32))
    if sampling == "afs":
        return AFSMultiVideoClipDataset(afs_candidates=afs_candidates, **kwargs)
    if sampling in ("uniform", "dense"):
        return MultiVideoClipDataset(**kwargs)
    raise ValueError(f"unsupported sampling: {sampling}")
