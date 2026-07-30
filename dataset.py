# -*- coding: utf-8 -*-
"""PyTorch Dataset：加载 CVAT 标注 + 图像，用于多任务分类"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from config import (
    DATA_ROOT,
    ANNOTATION_FILE,
    FRAMES_DIR,
    TASKS,
    IMAGE_SIZE,
)
from cvat_parser import parse_cvat_video_xml, get_task_indices_and_masks


def _pil_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise RuntimeError(f"无法加载图像: {path}") from e


def _cv2_loader(path):
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise RuntimeError(f"无法加载图像: {path}") from e


def default_collate(batch):
    """只保留有效样本的 batch，返回 dict。"""
    if not batch:
        return {}
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "image":
            out[k] = torch.stack([b[k] for b in batch])
        elif k == "valid_mask":
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


class ClassroomBehaviorDataset(Dataset):
    """
    课堂行为多任务数据集。
    每个样本是一帧图像 + 各任务的类别索引（未标注的为 -1）与有效 mask。
    """

    def __init__(
        self,
        data_root=None,
        annotation_file=None,
        frames_dir=None,
        frame_list=None,
        image_size=224,
        transform=None,
        is_train=True,
    ):
        self.data_root = Path(data_root or DATA_ROOT)
        self.annotation_file = annotation_file or ANNOTATION_FILE
        self.frames_dir = frames_dir or FRAMES_DIR
        self.image_size = image_size
        self.transform = transform
        self.is_train = is_train
        self.task_names = list(TASKS.keys())
        self.num_classes = {t: len(TASKS[t][0]) for t in self.task_names}

        if frame_list is not None:
            self.rows = frame_list
        else:
            xml_path = self.data_root / self.annotation_file
            frames_dir_abs = self.data_root / self.frames_dir
            if not xml_path.exists():
                # 尝试 annotations 子目录
                xml_path = self.data_root / "annotations" / self.annotation_file
            if not xml_path.exists():
                raise FileNotFoundError(f"请将 CVAT 导出的 {self.annotation_file} 放在: {self.data_root} 或 {self.data_root / 'annotations'}")
            frames = parse_cvat_video_xml(str(xml_path), frames_dir=str(frames_dir_abs))
            self.rows = get_task_indices_and_masks(frames, str(self.data_root))

        # 过滤掉图像路径不存在的（可选）
        self.valid_rows = []
        for r in self.rows:
            img_path = self.data_root / r["image_path"] if not os.path.isabs(r["image_path"]) else Path(r["image_path"])
            if not img_path.exists():
                img_path = self.data_root / self.frames_dir / Path(r["image_path"]).name
            r["_resolved_path"] = str(img_path)
            self.valid_rows.append(r)

    def __len__(self):
        return len(self.valid_rows)

    def __getitem__(self, idx):
        row = self.valid_rows[idx]
        path = row.get("_resolved_path") or (self.data_root / row["image_path"])
        if not os.path.isabs(path):
            path = self.data_root / path
        path = str(Path(path))

        try:
            img = _cv2_loader(path)
        except Exception:
            img = _pil_loader(path)

        if self.transform:
            try:
                out_t = self.transform(image=img)
                img = out_t["image"] if isinstance(out_t, dict) else out_t
            except TypeError:
                img = self.transform(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if isinstance(img, Image.Image):
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        out = {"image": img}
        valid_mask = []
        for t in self.task_names:
            idx_key = f"{t}_idx"
            valid_key = f"{t}_valid"
            out[idx_key] = torch.tensor(row.get(idx_key, -1), dtype=torch.long)
            out[valid_key] = torch.tensor(row.get(valid_key, False), dtype=torch.bool)
            valid_mask.append(row.get(valid_key, False))
        out["valid_mask"] = torch.tensor(valid_mask, dtype=torch.bool)  # [num_tasks]
        return out


def get_transforms(image_size, is_train=True):
    """简单 resize + 归一化；训练时加随机水平翻转。"""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        if is_train:
            t = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            t = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        return t
    except ImportError:
        pass

    # 无 albumentations 时用 torchvision
    import torchvision.transforms.functional as F
    from torchvision import transforms

    class SimpleTransform:
        def __init__(self, size, train):
            self.size = size
            self.image_size = size
            self.train = train
        def __call__(self, img):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = F.resize(img, [self.size, self.size])
            img = np.array(img)
            if self.train and np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            return torch.from_numpy(img).permute(2, 0, 1).float()

    return SimpleTransform(image_size, is_train)


def build_dataloaders(data_root, batch_size=16, image_size=224, train_ratio=0.8, num_workers=0, seed=42):
    """构建 train/val DataLoader。"""
    from torch.utils.data import DataLoader, Subset
    import random

    full_dataset = ClassroomBehaviorDataset(
        data_root=data_root,
        image_size=image_size,
        transform=None,
        is_train=True,
    )
    n = len(full_dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(n * train_ratio)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = ClassroomBehaviorDataset(
        data_root=data_root,
        image_size=image_size,
        frame_list=[full_dataset.valid_rows[i] for i in train_idx],
        transform=get_transforms(image_size, is_train=True),
        is_train=True,
    )
    val_dataset = ClassroomBehaviorDataset(
        data_root=data_root,
        image_size=image_size,
        frame_list=[full_dataset.valid_rows[i] for i in val_idx],
        transform=get_transforms(image_size, is_train=False),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=default_collate,
    )
    return train_loader, val_loader, full_dataset.task_names, full_dataset.num_classes
