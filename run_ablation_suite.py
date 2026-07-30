# -*- coding: utf-8 -*-
"""批量运行创新点与消融实验。

示例：
python run_ablation_suite.py --data_root /home/ma-user/work/test/data --device npu --amp --epochs 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], dry_run: bool = False):
    print("\n$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", default="./outputs/ablation_suite")
    ap.add_argument("--device", default="npu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=112)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    experiments = [
        ("i3d_baseline", "i3d", "none", False, "uniform", "none"),
        ("r3d_baseline", "r3d_18", "none", False, "uniform", "none"),
        ("videoswin_baseline", "swin3d_t", "none", False, "uniform", "none"),
        ("videoswin_bsf_mlp", "swin3d_t", "mlp", False, "uniform", "none"),
        ("videoswin_bsf_attn", "swin3d_t", "attn", False, "uniform", "none"),
        ("videoswin_wcls", "swin3d_t", "none", True, "uniform", "none"),
        ("videoswin_bsf_mlp_wcls", "swin3d_t", "mlp", True, "uniform", "none"),
        ("videoswin_afs", "swin3d_t", "none", False, "afs", "none"),
        ("videoswin_bsf_mlp_wcls_afs", "swin3d_t", "mlp", True, "afs", "none"),
        ("videoswin_ir_adapter", "swin3d_t", "none", False, "uniform", "ir_adapter"),
        ("videoswin_bsf_mlp_ir_adapter", "swin3d_t", "mlp", False, "uniform", "ir_adapter"),
        ("videoswin_full_ir_adapter", "swin3d_t", "mlp", True, "afs", "ir_adapter"),
        ("mvit_timesformer_proxy", "timesformer", "none", False, "uniform", "none"),
    ]

    for name, backbone, fusion, use_wcls, sampling, backbone_adapter in experiments:
        cmd = [
            sys.executable, "train_ablation.py",
            "--data_root", args.data_root,
            "--out_dir", str(Path(args.out_root) / name),
            "--backbone", backbone,
            "--fusion", fusion,
            "--sampling", sampling,
            "--backbone_adapter", backbone_adapter,
            "--folds", str(args.folds),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--clip_len", str(args.clip_len),
            "--stride", str(args.stride),
            "--image_size", str(args.image_size),
            "--device", args.device,
            "--discuss_loss_weight", "5.0",
        ]
        if args.amp:
            cmd.append("--amp")
        if use_wcls:
            cmd.append("--use_wcls")
        run(cmd, dry_run=args.dry_run)

    for backbone, fusion, adapter in [("swin3d_t", "none", "none"), ("swin3d_t", "none", "ir_adapter"), ("swin3d_t", "mlp", "ir_adapter"), ("swin3d_t", "attn", "none"), ("i3d", "none", "none")]:
        cmd = [
            sys.executable, "model_complexity.py",
            "--backbone", backbone,
            "--fusion", fusion,
            "--backbone_adapter", adapter,
            "--clip_len", str(args.clip_len),
            "--image_size", str(args.image_size),
            "--device", args.device,
            "--iters", "20",
        ]
        run(cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
