# -*- coding: utf-8 -*-
"""评估视频模型并输出论文常用结果（CSV/报告/混淆矩阵）。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from device_utils import get_device
from config import DATA_ROOT, TASKS, DEVICE
from video_dataset import build_video_dataloaders
from video_models import build_video_model


@torch.no_grad()
def collect_preds(model, loader, task_names, device):
    model.eval()
    preds = {t: [] for t in task_names}
    trues = {t: [] for t in task_names}
    for batch in loader:
        if not batch:
            continue
        video = batch["video"].to(device)
        logits = model(video)
        for t in task_names:
            valid = batch[f"{t}_valid"]
            if valid.sum() == 0:
                continue
            y = batch[f"{t}_idx"]
            p = logits[t].argmax(dim=1).cpu()
            mask = valid
            preds[t].append(p[mask])
            trues[t].append(y[mask])

    for t in task_names:
        if preds[t]:
            preds[t] = torch.cat(preds[t]).numpy()
            trues[t] = torch.cat(trues[t]).numpy()
        else:
            preds[t] = np.array([], dtype=np.int64)
            trues[t] = np.array([], dtype=np.int64)
    return preds, trues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=DATA_ROOT)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs/eval_video")
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    task_names = ckpt["task_names"]
    num_classes = ckpt["num_classes"]
    backbone = ckpt.get("backbone", "r3d_18")

    model = build_video_model(
        num_classes,
        backbone=backbone,
        pretrained=False,
        semantic_discuss_head=bool(ckpt.get("semantic_discuss_head", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, val_loader, _, _ = build_video_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        clip_len=int(ckpt.get("clip_len", 16)),
        stride=int(ckpt.get("stride", 8)),
        sample_rate=int(ckpt.get("sample_rate", 1)),
        image_size=int(ckpt.get("image_size", 112)),
        train_ratio=args.train_ratio,
        num_workers=0,
        seed=args.seed,
    )

    preds, trues = collect_preds(model, val_loader, task_names, device)

    rows = []
    for t in task_names:
        y_true = trues[t]
        y_pred = preds[t]
        if len(y_true) == 0:
            continue
        labels, _ = TASKS[t]
        acc = float((y_true == y_pred).mean())
        mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        rep = classification_report(
            y_true, y_pred, labels=list(range(len(labels))), target_names=labels, zero_division=0, digits=4
        )

        rows.append({"task": t, "accuracy": acc, "macro_f1": mf1, "n_samples": int(len(y_true))})

        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_matrix_{t}.csv")
        with open(out_dir / f"classification_report_{t}.txt", "w", encoding="utf-8") as f:
            f.write(f"Task: {t}\n\n{rep}\n")

    df = pd.DataFrame(rows).sort_values("task")
    df.to_csv(out_dir / "summary.csv", index=False)
    print("已输出到", out_dir)
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

